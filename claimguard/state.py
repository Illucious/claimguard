"""TASK 1 — State management.

An agent is a state machine. This TypedDict is the single memory object that
flows between every node of the LangGraph graph and is persisted by the
checkpointer after every step.

Each field has a *reducer* that tells LangGraph how to merge a node's partial
update into the existing state:
  - add_messages : appends messages (and supports RemoveMessage for compaction)
  - operator.add : appends lists / adds integers
  - merge_dicts  : shallow-merges dictionaries keyed by claim ID
  - (no reducer) : the latest value overwrites the previous one
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dicts(left: Optional[dict], right: Optional[dict]) -> dict:
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    # Conversation: Human / AI (incl. tool requests) / Tool result messages.
    messages: Annotated[list[AnyMessage], add_messages]

    # Task 7: running summary of compacted (deleted) older messages.
    summary: str
    compaction_count: Annotated[int, operator.add]

    # Task 2: reasoning steps taken in the current user turn (loop guard).
    step_count: int

    # Task 3/5/8: audit log of every JSON-RPC request/response, hook verdict and latency.
    tool_call_log: Annotated[list[dict[str, Any]], operator.add]
    error_count: Annotated[int, operator.add]

    # Task 5/6: structured business facts produced by the post-tool hook.
    risk_assessments: Annotated[dict[str, dict], merge_dicts]   # claim_id -> risk record

    # Task 6: Human-in-the-loop.
    pending_review: Optional[dict]                               # set → graph routes to human_review
    human_decisions: Annotated[dict[str, dict], merge_dicts]     # claim_id -> {decision, comment, reviewer, at}
