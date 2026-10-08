"""The LangGraph state machine.

        START
          │ route_compaction
          ├──────────────► summarize ─┐            (Task 7)
          ▼                           │
      reasoning ◄─────────────────────┘            (Task 2)
          │ route_after_reasoning
          ├── no tool calls ──► END
          ▼
        tools                                      (Task 3/4/5)
          │ route_after_tools
          ├── pending_review ──► human_review ──► (route_compaction) …   (Task 6)
          ├── too many messages ──► summarize
          └──► reasoning
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .compaction import make_summarize_node, route_compaction
from .config import Settings, settings as default_settings
from .hitl import human_review_node, route_after_tools
from .llm import create_chat_model
from .reasoning import make_reasoning_node, route_after_reasoning
from .state import AgentState
from .telemetry import Telemetry, telemetry as default_telemetry
from .tool_executor import make_tool_node
from .tool_provider import BaseToolProvider


def build_graph(provider: BaseToolProvider, *, checkpointer=None, llm=None, summarizer_llm=None,
                telemetry: Telemetry | None = None, cfg: Settings | None = None):
    cfg = cfg or default_settings
    telemetry = telemetry or default_telemetry

    # Dynamically discovered schemas (MCP tools/list or local) are bound to the LLM here.
    llm = llm or create_chat_model("agent", cfg)
    bound_llm = llm.bind_tools([t.as_openai_tool() for t in provider.tools])
    summarizer_llm = summarizer_llm or create_chat_model("summary", cfg)

    graph = StateGraph(AgentState)
    graph.add_node("summarize", make_summarize_node(summarizer_llm))
    graph.add_node("reasoning", make_reasoning_node(bound_llm, cfg))
    graph.add_node("tools", make_tool_node(provider, telemetry))
    graph.add_node("human_review", human_review_node)

    graph.add_conditional_edges(START, route_compaction, {"summarize": "summarize", "reasoning": "reasoning"})
    graph.add_edge("summarize", "reasoning")
    graph.add_conditional_edges("reasoning", route_after_reasoning, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools,
                                {"human_review": "human_review", "summarize": "summarize", "reasoning": "reasoning"})
    graph.add_conditional_edges("human_review", route_compaction, {"summarize": "summarize", "reasoning": "reasoning"})

    return graph.compile(checkpointer=checkpointer)
