"""TASK 7 — Context compaction (checkpointing lives in agent.py).

If len(messages) > COMPACTION_THRESHOLD (10), the `summarize` node makes a
separate, tool-less ("offline") LLM call that summarizes the first
COMPACTION_BATCH (8) messages into one paragraph, deletes them with
RemoveMessage and stores the paragraph in state["summary"]. The reasoning node
injects that summary into the system prompt.

Safety detail: a ToolMessage must always follow the AIMessage that requested
it. The cut point is moved so we never leave an orphaned ToolMessage at the
start of the remaining history (providers reject that).
"""
from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from .config import settings
from .llm import message_text
from .prompts import SUMMARY_PROMPT
from .state import AgentState


def should_compact(state: AgentState) -> bool:
    return len(state.get("messages", [])) > settings.compaction_threshold


def route_compaction(state: AgentState) -> str:
    return "summarize" if should_compact(state) else "reasoning"


def safe_cut_index(messages: list, batch: int) -> int:
    n = len(messages)
    cut = min(batch, n - 1)
    while cut < n and isinstance(messages[cut], ToolMessage):
        cut += 1
    if cut >= n:   # nothing would remain → move the cut backwards instead
        cut = min(batch, n - 1)
        while cut > 0 and isinstance(messages[cut], ToolMessage):
            cut -= 1
    return cut


def _transcript(messages: list) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "USER"
        elif isinstance(m, AIMessage):
            role = "ASSISTANT"
        elif isinstance(m, ToolMessage):
            role = f"TOOL[{m.name}]"
        else:
            role = m.type.upper()
        text = message_text(m)[:1500]
        if isinstance(m, AIMessage) and m.tool_calls:
            text += " CALLS: " + "; ".join(f"{tc['name']}({tc.get('args')})" for tc in m.tool_calls)
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def make_summarize_node(summarizer_llm):
    async def summarize_node(state: AgentState, config: RunnableConfig) -> dict:
        messages = state.get("messages", [])
        cut = safe_cut_index(messages, settings.compaction_batch)
        if cut <= 0:
            return {}
        old = messages[:cut]
        previous = state.get("summary", "")
        prompt = (f"Existing summary:\n{previous}\n\n" if previous else "") + "New transcript to fold in:\n" + _transcript(old)

        try:
            result = await summarizer_llm.ainvoke([SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=prompt)],
                                                  config=config)
            summary = message_text(result).strip()
        except Exception as exc:  # noqa: BLE001 — never lose context: fall back to a raw truncated transcript
            summary = (previous + " " + _transcript(old))[-3000:] + f" [summarizer failed: {type(exc).__name__}]"

        return {
            "messages": [RemoveMessage(id=m.id) for m in old],
            "summary": summary,
            "compaction_count": 1,
            "tool_call_log": [{"type": "compaction", "removed_messages": len(old),
                               "remaining_messages": len(messages) - len(old),
                               "summary": summary, "ts": datetime.now().isoformat()}],
        }

    return summarize_node
