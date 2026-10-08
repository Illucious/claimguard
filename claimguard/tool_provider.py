"""TASK 3 + TASK 4 — Where tools come from.

Both providers expose the same interface:
    await provider.start()                 -> discovers tools (ToolSpec = name + description + JSON schema)
    provider.tools                         -> list[ToolSpec]  (bound to the LLM)
    await provider.call_tool(name, args)   -> dict           (executes the tool)

LocalToolProvider (TOOL_MODE=local)  — Task 3
    Python functions are wrapped into JSON schemas in-process and called directly.

MCPToolProvider  (TOOL_MODE=mcp, default) — Task 4
    The agent knows NOTHING about the tools at build time. On startup it connects
    to the MCP server (spawned over stdio, or via HTTP if MCP_SERVER_URL is set),
    sends `tools/list`, converts the returned schemas for the LLM and routes every
    call as `tools/call` over the transport.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from .config import PROJECT_ROOT, Settings

log = logging.getLogger("claimguard.tools")


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict

    def as_openai_tool(self) -> dict:
        """JSON schema format accepted by ChatGroq / ChatGoogleGenerativeAI .bind_tools()."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": clean_schema(self.input_schema)}}


def clean_schema(schema: dict) -> dict:
    """Make auto-generated schemas portable across providers:
    drop 'title' keys and collapse Optional[X] (anyOf [X, null]) into X."""
    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: _walk(v) for k, v in node.items() if k != "title"}
            any_of = node.get("anyOf")
            if isinstance(any_of, list):
                non_null = [s for s in any_of if not (isinstance(s, dict) and s.get("type") == "null")]
                if len(non_null) == 1 and len(any_of) == 2:
                    node.pop("anyOf")
                    node = {**non_null[0], **node}
            return node
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    cleaned = _walk(copy.deepcopy(schema or {}))
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    return cleaned


class BaseToolProvider:
    mode = "base"

    def __init__(self) -> None:
        self.tools: list[ToolSpec] = []

    @property
    def registry(self) -> dict[str, ToolSpec]:
        return {t.name: t for t in self.tools}

    async def start(self) -> None: ...
    async def call_tool(self, name: str, args: dict) -> dict: ...
    async def close(self) -> None: ...


# =============================================================== TASK 3: local
class LocalToolProvider(BaseToolProvider):
    mode = "local"

    async def start(self) -> None:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        from insurance_tools.tools import ALL_TOOLS

        self._functions = {fn.__name__: fn for fn in ALL_TOOLS}
        self.tools = []
        for fn in ALL_TOOLS:
            schema = convert_to_openai_tool(fn)["function"]   # Python function -> JSON schema
            self.tools.append(ToolSpec(schema["name"], schema.get("description", ""), schema.get("parameters", {})))
        log.info("Local tools registered: %s", [t.name for t in self.tools])

    async def call_tool(self, name: str, args: dict) -> dict:
        fn = self._functions[name]
        return await asyncio.to_thread(fn, **args)   # route payload to the Python function


# ================================================================= TASK 4: MCP
class MCPToolProvider(BaseToolProvider):
    mode = "mcp"

    def __init__(self, server_url: str = "") -> None:
        super().__init__()
        self.server_url = server_url
        self.session = None
        self._stack: AsyncExitStack | None = None

    async def start(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        if self.server_url:   # remote MCP server over Streamable HTTP
            from mcp.client.streamable_http import streamablehttp_client

            read, write, _ = await self._stack.enter_async_context(streamablehttp_client(self.server_url))
            transport = f"http ({self.server_url})"
        else:                 # spawn local MCP server as a subprocess, talk over stdio
            params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"],
                                           cwd=str(PROJECT_ROOT), env=dict(os.environ))
            read, write = await self._stack.enter_async_context(stdio_client(params))
            transport = "stdio"

        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        await self.refresh_tools()
        log.info("MCP connected via %s; discovered %d tools", transport, len(self.tools))

    async def refresh_tools(self) -> None:
        """JSON-RPC `tools/list` → dynamic schemas for the LLM."""
        result = await self.session.list_tools()
        self.tools = [ToolSpec(t.name, t.description or "", dict(t.inputSchema or {})) for t in result.tools]

    async def call_tool(self, name: str, args: dict) -> dict:
        """JSON-RPC `tools/call` over the MCP transport."""
        result = await self.session.call_tool(name, arguments=args)
        texts = [c.text for c in (result.content or []) if getattr(c, "type", "") == "text"]
        if getattr(result, "isError", False):
            return {"error": {"code": 500, "message": " ".join(texts) or "MCP tool execution error", "hint": ""}}
        if texts:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return {"text": texts[0]}
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured.get("result", structured) if set(structured) == {"result"} else structured
        return {"error": {"code": 502, "message": "MCP tool returned no content", "hint": ""}}

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()
            self._stack = None


def make_tool_provider(cfg: Settings) -> BaseToolProvider:
    if cfg.tool_mode == "local":
        return LocalToolProvider()
    return MCPToolProvider(server_url=cfg.mcp_server_url)
