"""TASK 3 (execution) — the tools node.

For every tool call the reasoning node produced:
    JSON-RPC request
      → pre_tool_hook   (Task 5)  — invalid? return 400/403 to the LLM, skip execution
      → provider.call_tool        — local Python function (Task 3) or MCP tools/call (Task 4)
      → post_tool_hook  (Task 5)  — validate business invariants, mask PII, detect HITL need
      → ToolMessage appended to state["messages"]
Every step is wrapped in a Langfuse 'tool' observation (Task 8) and logged to state["tool_call_log"].
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from .hooks import post_tool_hook, pre_tool_hook
from .reasoning import to_jsonrpc_request
from .state import AgentState
from .telemetry import Telemetry
from .tool_provider import BaseToolProvider


def make_tool_node(provider: BaseToolProvider, telemetry: Telemetry):
    async def tool_node(state: AgentState, config: RunnableConfig) -> dict:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        registry = provider.registry
        risk_view = dict(state.get("risk_assessments") or {})   # updated as we go, so later calls see earlier results
        decisions = state.get("human_decisions") or {}

        messages, events, new_risk = [], [], {}
        errors, pending = 0, None

        for tc in tool_calls:
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            name, args = tc["name"], tc.get("args") or {}
            request = to_jsonrpc_request({**tc, "id": call_id})
            started = time.perf_counter()

            with telemetry.span(f"tool:{name}", input=request, as_type="tool") as span:
                pre = pre_tool_hook(call_id, name, args, registry, risk_view, decisions)
                if not pre.ok:
                    response, stage = pre.response, "pre_tool_hook"
                    if pre.review_request and pre.review_request["claim_id"] not in decisions and pending is None:
                        pending = pre.review_request
                else:
                    try:
                        raw = await provider.call_tool(name, pre.args)
                    except Exception as exc:  # noqa: BLE001 — transport/server failure
                        raw = {"error": {"code": 502, "message": f"Tool server failure: {exc}", "hint": "Retry once."}}
                    post = post_tool_hook(call_id, name, pre.args, raw)
                    response = post.response
                    stage = "ok" if post.ok else response["error"]["data"]["stage"]
                    if post.risk_record:
                        risk_view[post.risk_record["claim_id"]] = post.risk_record
                        new_risk[post.risk_record["claim_id"]] = post.risk_record
                    if post.review_request and post.review_request["claim_id"] not in decisions and pending is None:
                        pending = post.review_request

                latency_ms = round((time.perf_counter() - started) * 1000, 1)
                is_error = "error" in response
                if span is not None:
                    span.update(output=response, level="ERROR" if is_error else "DEFAULT",
                                status_message=response["error"]["message"] if is_error else None,
                                metadata={"stage": stage, "latency_ms": latency_ms, "transport": provider.mode})

            errors += int(is_error)
            messages.append(ToolMessage(content=json.dumps(response, ensure_ascii=False, default=str),
                                        tool_call_id=call_id, name=name, status="error" if is_error else "success"))
            events.append({"type": "tool_response", "tool": name, "stage": stage, "latency_ms": latency_ms,
                           "request": request, "response": response, "ts": datetime.now().isoformat()})

        update: dict = {"messages": messages, "tool_call_log": events, "error_count": errors}
        if new_risk:
            update["risk_assessments"] = new_risk
        if pending:
            update["pending_review"] = pending
            update["tool_call_log"] = events + [{"type": "hitl_triggered", "review": pending,
                                                 "ts": datetime.now().isoformat()}]
        return update

    return tool_node
