"""TASK 2 — The reasoning node (the engine).

This node NEVER executes anything. It sends the state to the tool-bound LLM and
returns exactly one of:
  * an AIMessage containing tool_calls  → logged as JSON-RPC `tools/call` requests
  * an AIMessage containing final text  → the graph ends
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from .config import Settings
from .llm import message_text
from .prompts import build_system_prompt
from .state import AgentState

log = logging.getLogger("claimguard.reasoning")


def to_jsonrpc_request(tool_call: dict) -> dict:
    """Represent an LLM tool call as an MCP / JSON-RPC 2.0 request."""
    return {
        "jsonrpc": "2.0",
        "id": tool_call.get("id"),
        "method": "tools/call",
        "params": {"name": tool_call["name"], "arguments": tool_call.get("args", {})},
    }


def make_reasoning_node(bound_llm, cfg: Settings):
    async def reasoning_node(state: AgentState, config: RunnableConfig) -> dict:
        step = state.get("step_count", 0)

        # Loop guard: never let a confused model spin forever.
        if step >= cfg.max_steps:
            msg = AIMessage(content=(
                "I stopped because this request needed more than "
                f"{cfg.max_steps} reasoning steps. Please narrow the request (for example, one claim at a time)."))
            return {"messages": [msg], "step_count": step + 1,
                    "tool_call_log": [{"type": "loop_guard", "step": step, "ts": datetime.now().isoformat()}]}

        system = SystemMessage(content=build_system_prompt(
            state.get("summary", ""), state.get("human_decisions"), cfg.high_value_threshold))
        history = list(state.get("messages", []))
        # After compaction the history may not start with a user turn; some providers require one.
        if not history or not isinstance(history[0], HumanMessage):
            history = [HumanMessage(content="(Earlier conversation is summarized in the system prompt.)")] + history

        started = time.perf_counter()
        response = None
        last_error = None
        for attempt in range(2):  # one retry for transient / malformed tool-call errors from the provider
            try:
                response = await bound_llm.ainvoke([system] + history, config=config)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning("LLM call failed (attempt %s): %s", attempt + 1, exc)
        if response is None:
            response = AIMessage(content=f"Sorry, the language model is unavailable right now ({type(last_error).__name__}). Please retry.")

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        usage = getattr(response, "usage_metadata", None) or {}
        events = []
        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                events.append({"type": "llm_tool_request", "step": step, "latency_ms": latency_ms,
                               "tokens": usage, "ts": datetime.now().isoformat(), **to_jsonrpc_request(tc)})
        else:
            events.append({"type": "final_answer", "step": step, "latency_ms": latency_ms, "tokens": usage,
                           "ts": datetime.now().isoformat(), "content": message_text(response)[:2000]})

        return {"messages": [response], "step_count": step + 1, "tool_call_log": events}

    return reasoning_node


def route_after_reasoning(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END
