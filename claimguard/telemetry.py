"""TASK 8 — Continuous telemetry with Langfuse (Python SDK v3, OpenTelemetry based).

What ends up in a Langfuse trace for ONE user request:
  claimguard-agent  (agent observation = the whole LangGraph run, input/output, session = thread_id)
  ├─ LangGraph node spans + every LLM generation      ← via the LangChain CallbackHandler
  │     (model, prompt, completion, token usage, latency)
  ├─ tool:<name>   (tool observation)                  ← created in tool_executor.py
  │     input  = JSON-RPC tools/call request
  │     output = JSON-RPC response (result or 400/403/404/422 error)
  │     metadata = hook stage, latency_ms; level=ERROR when the call failed
  └─ human_review / compaction events

If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set, every method is a
no-op so the agent still runs locally.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

log = logging.getLogger("claimguard.telemetry")

try:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler
    _SDK_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    log.warning("Langfuse SDK unavailable (%s); telemetry disabled.", exc)
    _SDK_AVAILABLE = False


class Telemetry:
    def __init__(self) -> None:
        self.enabled = bool(_SDK_AVAILABLE and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
        self.client = get_client() if self.enabled else None
        if self.enabled:
            log.info("Langfuse telemetry enabled (host=%s)", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))

    def callbacks(self) -> list:
        """LangChain callback → captures every LLM call (tokens, latency, prompts) and every graph node."""
        return [CallbackHandler()] if self.enabled else []

    @contextmanager
    def trace(self, name: str, *, input: Any, session_id: str, user_id: str | None = None,
              metadata: dict | None = None) -> Iterator[Optional[Any]]:
        """Root observation that wraps one full LangGraph execution loop."""
        if not self.enabled:
            yield None
            return
        with self.client.start_as_current_observation(name=name, as_type="agent", input=input,
                                                      metadata=metadata or {}) as span:
            span.update_trace(name=name, session_id=session_id, user_id=user_id, input=input,
                              tags=["claimguard", os.getenv("TOOL_MODE", "mcp")])
            yield span

    @contextmanager
    def span(self, name: str, *, input: Any = None, metadata: dict | None = None,
             as_type: str = "span") -> Iterator[Optional[Any]]:
        if not self.enabled:
            yield None
            return
        with self.client.start_as_current_observation(name=name, as_type=as_type, input=input,
                                                      metadata=metadata or {}) as span:
            yield span

    def current_trace_id(self) -> Optional[str]:
        try:
            return self.client.get_current_trace_id() if self.enabled else None
        except Exception:  # noqa: BLE001
            return None

    def trace_url(self, trace_id: str | None) -> Optional[str]:
        try:
            return self.client.get_trace_url(trace_id=trace_id) if (self.enabled and trace_id) else None
        except Exception:  # noqa: BLE001
            return None

    def flush(self) -> None:
        if self.enabled:
            try:
                self.client.flush()
            except Exception as exc:  # noqa: BLE001
                log.warning("Langfuse flush failed: %s", exc)


telemetry = Telemetry()
