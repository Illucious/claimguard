"""End-to-end graph test with a scripted (fake) LLM and the REAL MCP server over stdio.

Verifies, without any API key:
  Task 2/3/4  tool requests are discovered via MCP tools/list and executed via tools/call
  Task 5      bad ID → 400 → corrected retry
  Task 6      HIGH-risk claim pauses the graph; decision resumes it
  Task 7      paused state survives an agent "restart"; compaction triggers when > 10 messages
"""
import csv

from langchain_core.messages import AIMessage

from claimguard.agent import ClaimGuardAgent
from claimguard.config import Settings
from claimguard.tool_provider import MCPToolProvider


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages, config=None):
        return self.script.pop(0)


def call(name, args, cid):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


async def test_full_assessment_with_hitl_restart_and_compaction(temp_data):
    cfg = Settings()  # picks up temp DATA_DIR / CHECKPOINT_DB from env
    summarizer = ScriptedLLM([AIMessage(content="User asked to assess C1003 (theft, ₹4,50,000, policy P1003). "
                                                "Risk HIGH, score 80, sent to human review.")])
    script = [
        AIMessage(content="", tool_calls=[call("get_claim_details", {"claim_id": "c1003"}, "a1")]),   # → 400
        AIMessage(content="", tool_calls=[call("get_claim_details", {"claim_id": "C1003"}, "a2"),
                                          call("get_policy_details", {"policy_id": "P1003"}, "a3"),
                                          call("get_claim_history", {"policy_id": "P1003"}, "a4")]),
        AIMessage(content="", tool_calls=[call("check_claim_eligibility", {"claim_id": "C1003"}, "a5"),
                                          call("calculate_estimated_payout", {"claim_id": "C1003"}, "a6"),
                                          call("assess_claim_risk", {"claim_id": "C1003"}, "a7")]),   # → HITL
    ]
    after_resume = [
        AIMessage(content="", tool_calls=[call("update_claim_status",
                                               {"claim_id": "C1003", "new_status": "Rejected",
                                                "reason": "Rejected by reviewer: suspicious history"}, "a8")]),
        AIMessage(content="Claim C1003 has been rejected by the human reviewer."),
    ]

    # ---- run 1: should pause for human review
    agent = await ClaimGuardAgent(cfg, provider=MCPToolProvider(), llm=ScriptedLLM(script),
                                  summarizer_llm=summarizer).start()
    assert len(agent.provider.tools) == 9                      # discovered via MCP tools/list
    r1 = await agent.chat("t1", "Assess claim c1003")
    assert r1["status"] == "awaiting_human_review"
    assert r1["review"]["claim_id"] == "C1003" and r1["review"]["risk_level"] == "HIGH"
    first = r1["events"][1]
    assert first["type"] == "tool_response" and first["response"]["error"]["code"] == 400
    await agent.stop()                                          # simulate server shutdown

    # ---- run 2: brand-new agent process, same checkpoint DB → resume works
    agent2 = await ClaimGuardAgent(cfg, provider=MCPToolProvider(), llm=ScriptedLLM(after_resume),
                                   summarizer_llm=summarizer).start()
    state = await agent2.get_thread("t1")
    assert state["pending_review"]["claim_id"] == "C1003"
    r2 = await agent2.resume("t1", "reject", "Suspicious claim history")
    assert r2["status"] == "completed" and "rejected" in r2["answer"].lower()
    assert r2["compacted"] is True                              # 11 messages > 10 → summarize node ran

    final = await agent2.get_thread("t1")
    assert final["human_decisions"]["C1003"]["decision"] == "reject"
    assert final["summary"] and final["compaction_count"] == 1
    await agent2.stop()

    with open(temp_data / "claims.csv", newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["claim_id"] == "C1003")
    assert row["claim_status"] == "Rejected"
    assert (temp_data / "traces" / "traces.jsonl").exists()


async def test_simple_question_local_mode(temp_data):
    from claimguard.tool_provider import LocalToolProvider

    cfg = Settings()
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[call("get_policy_details", {"policy_id": "P1003"}, "b1")]),
        AIMessage(content="The annual premium for P1003 is ₹24,500."),
    ])
    agent = await ClaimGuardAgent(cfg, provider=LocalToolProvider(), llm=llm,
                                  summarizer_llm=ScriptedLLM([])).start()
    r = await agent.chat("t2", "What is the premium for P1003?")
    assert r["status"] == "completed" and r["tools_used"] == ["get_policy_details"]
    assert r["errors_in_run"] == 0
    await agent.stop()
