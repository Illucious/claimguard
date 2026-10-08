"""TASK 4 — Standard MCP server exposing the insurance tools.

The agent does NOT import these tools. It starts this server as a subprocess
(stdio transport) or connects over HTTP, sends `tools/list` to discover the
schemas, and sends `tools/call` to execute them.

Run standalone:
    python -m mcp_server.server                 # stdio (spawned by the agent)
    MCP_TRANSPORT=streamable-http python -m mcp_server.server   # HTTP on :8001/mcp
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from insurance_tools.tools import ALL_TOOLS  # noqa: E402

# stdout is reserved for the JSON-RPC protocol in stdio mode → log to stderr only.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

mcp = FastMCP(
    "claimguard-insurance-tools",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
)

for fn in ALL_TOOLS:
    mcp.tool()(fn)  # schema generated from type hints + docstring


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
