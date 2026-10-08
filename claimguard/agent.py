"""ClaimGuardAgent — the runtime used by the CLI, the FastAPI app and the eval scripts.

Responsibilities:
  * start the tool provider (MCP client → tools/list)                       Task 4
  * open the SQLite checkpointer so every step is persisted to disk          Task 7
  * wrap each graph execution in a Langfuse trace                            Task 8
  * detect the HITL interrupt and resume with Command(resume=...)            Task 6
  * save a local copy of every run's trace for the LLM judge                 Task 9
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from .config import Settings, settings as default_settings
from .graph import build_graph
from .hitl import normalize_decision
from .llm import message_text
from .telemetry import Telemetry, telemetry as default_telemetry
from .tool_provider import BaseToolProvider, make_tool_provider

log = logging.getLogger("claimguard.agent")


class ReviewNotPendingError(Exception):
    pass


class ThreadAwaitingReviewError(Exception):
    pass


class ClaimGuardAgent:
    def __init__(self, cfg: Settings | None = None, *, provider: BaseToolProvider | None = None,
                 llm=None, summarizer_llm=None, telemetry: Telemetry | None = None) -> None:
        self.cfg = cfg or default_settings
        self.provider = provider or make_tool_provider(self.cfg)
        self.telemetry = telemetry or default_telemetry
        self._llm, self._summarizer = llm, summarizer_llm
        self._stack: AsyncExitStack | None = None
        self.graph = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> "ClaimGuardAgent":
        self._stack = AsyncExitStack()
        await self.provider.start()
        self._stack.push_async_callback(self.provider.close)

        self.cfg.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = await self._stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(self.cfg.checkpoint_db)))

        self.graph = build_graph(self.provider, checkpointer=checkpointer, llm=self._llm,
                                 summarizer_llm=self._summarizer, telemetry=self.telemetry, cfg=self.cfg)
        log.info("ClaimGuard ready: %d tools via %s, checkpoints at %s",
                 len(self.provider.tools), self.provider.mode, self.cfg.checkpoint_db)
        return self

    async def stop(self) -> None:
        self.telemetry.flush()
        if self._stack:
            await self._stack.aclose()
            self._stack = None

    # --------------------------------------------------------------- public
    async def chat(self, thread_id: str | None, message: str, user_id: str | None = None) -> dict:
        thread_id = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        config = self._config(thread_id)
        if await self._pending_interrupt(config):
            raise ThreadAwaitingReviewError(
                f"Thread {thread_id} is waiting for a human review decision. Submit it before sending new messages.")
        payload = {"messages": [HumanMessage(content=message)], "step_count": 0}
        return await self._run(thread_id, config, payload, user_input=message, user_id=user_id, kind="chat")

    async def resume(self, thread_id: str, decision: str, comment: str = "", reviewer: str = "human-reviewer") -> dict:
        normalized = normalize_decision(decision)
        if normalized is None:
            raise ValueError("decision must be 'approve' or 'reject'")
        config = self._config(thread_id)
        if not await self._pending_interrupt(config):
            raise ReviewNotPendingError(f"Thread {thread_id} has no pending human review.")
        command = Command(resume={"decision": normalized, "comment": comment, "reviewer": reviewer})
        return await self._run(thread_id, config, command, user_input=f"[review] {normalized}: {comment}",
                               user_id=reviewer, kind="review")

    async def get_thread(self, thread_id: str) -> dict:
        config = self._config(thread_id)
        snap = await self.graph.aget_state(config)
        values = snap.values or {}
        return {
            "thread_id": thread_id,
            "exists": bool(values),
            "message_count": len(values.get("messages", [])),
            "summary": values.get("summary", ""),
            "compaction_count": values.get("compaction_count", 0),
            "error_count": values.get("error_count", 0),
            "risk_assessments": values.get("risk_assessments", {}),
            "human_decisions": values.get("human_decisions", {}),
            "pending_review": await self._pending_interrupt(config),
            "next_nodes": list(snap.next or []),
            "messages": [{"type": m.type, "content": message_text(m)[:500],
                          "tool_calls": [tc["name"] for tc in getattr(m, "tool_calls", []) or []]}
                         for m in values.get("messages", [])],
        }

    # -------------------------------------------------------------- helpers
    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": self.cfg.recursion_limit,
                "callbacks": self.telemetry.callbacks(),
                "metadata": {"langfuse_session_id": thread_id}}

    async def _pending_interrupt(self, config) -> Optional[dict]:
        snap = await self.graph.aget_state(config)
        for task in snap.tasks or []:
            for intr in getattr(task, "interrupts", None) or []:
                return intr.value
        return None

    async def _run(self, thread_id: str, config: dict, payload: Any, *, user_input: str,
                   user_id: str | None, kind: str) -> dict:
        before = await self.graph.aget_state(config)
        log_start = len((before.values or {}).get("tool_call_log", []))
        msg_count_before = len((before.values or {}).get("messages", []))
        trace_id = None

        with self.telemetry.trace("claimguard-agent", input={"message": user_input, "kind": kind},
                                  session_id=thread_id, user_id=user_id,
                                  metadata={"tool_mode": self.provider.mode}) as span:
            trace_id = self.telemetry.current_trace_id()
            error = None
            try:
                await self.graph.ainvoke(payload, config)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                log.exception("Graph execution failed")
            result = await self._collect(thread_id, config, log_start, msg_count_before)
            if error:
                result.update(status="error", error=error)
            result["trace_id"] = trace_id
            result["trace_url"] = self.telemetry.trace_url(trace_id)
            if span is not None:
                span.update(output={"status": result["status"], "answer": result["answer"],
                                    "review": result["review"]},
                            level="ERROR" if error else "DEFAULT")
        self.telemetry.flush()
        self._save_local_trace(thread_id, kind, user_input, result)
        return result

    async def _collect(self, thread_id: str, config: dict, log_start: int, msg_count_before: int) -> dict:
        snap = await self.graph.aget_state(config)
        values = snap.values or {}
        events = values.get("tool_call_log", [])[log_start:]
        review = await self._pending_interrupt(config)

        answer = ""
        for m in reversed(values.get("messages", [])):
            if isinstance(m, AIMessage) and not m.tool_calls and message_text(m).strip():
                answer = message_text(m)
                break
        status = "awaiting_human_review" if review else "completed"
        if review:
            answer = (f"⏸ Claim {review.get('claim_id')} requires human approval "
                      f"({review.get('trigger')}). Waiting for an approve/reject decision.")
        tools_used = [e["tool"] for e in events if e.get("type") == "tool_response"]
        return {
            "thread_id": thread_id,
            "status": status,
            "answer": answer,
            "review": review,
            "tools_used": tools_used,
            "errors_in_run": sum(1 for e in events if e.get("type") == "tool_response" and "error" in e["response"]),
            "compacted": any(e.get("type") == "compaction" for e in events),
            "events": events,
        }

    def _save_local_trace(self, thread_id: str, kind: str, user_input: str, result: dict) -> None:
        """Local JSONL trace store — lets the LLM judge run even without Langfuse."""
        try:
            self.cfg.trace_file.parent.mkdir(parents=True, exist_ok=True)
            record = {"trace_id": result.get("trace_id") or uuid.uuid4().hex, "thread_id": thread_id,
                      "kind": kind, "timestamp": datetime.now().isoformat(timespec="seconds"),
                      "user_input": user_input, "status": result["status"], "final_answer": result["answer"],
                      "events": result["events"]}
            with self.cfg.trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save local trace: %s", exc)
