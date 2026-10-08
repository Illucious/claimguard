"""TASK 6 — Human-in-the-Loop routing.

Flow:
  tools node ──(conditional edge: route_after_tools)──► human_review
                                                         │  interrupt(...)  ← graph PAUSES here;
                                                         │  state is persisted by the checkpointer
  CLI prompt / POST /webhook/review sends Command(resume={"decision": "approve"|"reject", ...})
                                                         ▼
                              decision stored in state, message added → back to reasoning

Because the paused state lives in the SQLite checkpoint, the approval can arrive
minutes later — even after the server restarts.
"""
from __future__ import annotations

from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from .compaction import route_compaction
from .state import AgentState

VALID_DECISIONS = {"approve": "approve", "approved": "approve", "reject": "reject", "rejected": "reject"}


def normalize_decision(value) -> str | None:
    if isinstance(value, dict):
        value = value.get("decision")
    return VALID_DECISIONS.get(str(value or "").strip().lower())


def human_review_node(state: AgentState) -> dict:
    review = state.get("pending_review") or {}
    claim_id = review.get("claim_id", "UNKNOWN")

    # ---- PAUSE. Everything before this line re-runs on resume, so it must be side-effect free.
    response = interrupt({
        "type": "human_review_required",
        **review,
        "question": f"Claim {claim_id} needs human approval. Reply 'approve' or 'reject' (optionally with a comment).",
    })

    decision = normalize_decision(response)
    if decision is None:
        raise ValueError(f"Invalid review decision {response!r}; expected 'approve' or 'reject'.")
    comment = (response.get("comment") if isinstance(response, dict) else "") or "No comment provided."
    reviewer = (response.get("reviewer") if isinstance(response, dict) else None) or "human-reviewer"

    record = {"decision": decision, "comment": comment, "reviewer": reviewer,
              "at": datetime.now().isoformat(timespec="seconds"), "review": review}
    message = HumanMessage(content=(
        f"[HUMAN REVIEWER DECISION] Claim {claim_id}: {decision.upper()}. "
        f"Reviewer: {reviewer}. Comment: {comment}\n"
        f"Record this decision with update_claim_status and then give the final summary to the user."))
    return {
        "messages": [message],
        "human_decisions": {claim_id: record},
        "pending_review": None,
        "tool_call_log": [{"type": "human_decision", "claim_id": claim_id, **record}],
    }


def route_after_tools(state: AgentState) -> str:
    """Conditional edge out of the tools node."""
    pending = state.get("pending_review")
    if pending and pending.get("claim_id") not in (state.get("human_decisions") or {}):
        return "human_review"
    return route_compaction(state)
