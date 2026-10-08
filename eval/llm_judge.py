"""TASK 9 (eval) — LLM-as-a-judge for "Error Recovery".

Feeds the 10 most recent historical traces to an LLM and asks it to grade each
one 1-5 for error recovery.

    python -m eval.llm_judge                     # auto: Langfuse if configured, else local traces
    python -m eval.llm_judge --source local --limit 10
    python -m eval.llm_judge --source langfuse

Outputs eval/results/report.json and eval/results/report.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from claimguard.config import settings
from claimguard.llm import create_chat_model, message_text
from claimguard.prompts import JUDGE_PROMPT

RESULTS_DIR = Path(__file__).with_name("results")


# ------------------------------------------------------------------ trace loading
def load_local_traces(limit: int) -> list[dict]:
    """Group runs by thread (a HITL scenario = chat run + review run) and keep the latest `limit`."""
    path = settings.trace_file
    if not path.exists():
        return []
    threads: "OrderedDict[str, dict]" = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        t = threads.setdefault(rec["thread_id"], {"trace_id": rec["trace_id"], "thread_id": rec["thread_id"],
                                                 "user_input": rec["user_input"], "steps": [], "final_answer": ""})
        t["steps"].append({"kind": rec["kind"], "input": rec["user_input"], "events": rec["events"]})
        t["final_answer"] = rec["final_answer"]
        threads.move_to_end(rec["thread_id"])
    return list(threads.values())[-limit:]


def load_langfuse_traces(limit: int) -> list[dict]:
    from langfuse import get_client

    client = get_client()
    listing = client.api.trace.list(limit=limit * 2, name="claimguard-agent", order_by="timestamp.desc")
    sessions: "OrderedDict[str, dict]" = OrderedDict()
    for summary in reversed(listing.data):  # oldest → newest
        full = client.api.trace.get(summary.id)
        events = []
        for obs in sorted(full.observations, key=lambda o: o.start_time):
            if obs.name and (obs.name.startswith("tool:") or obs.type == "GENERATION"):
                events.append({"type": obs.type, "name": obs.name, "level": obs.level,
                               "status_message": obs.status_message,
                               "input": obs.input if obs.name.startswith("tool:") else None,
                               "output": obs.output if obs.name.startswith("tool:") else _short(obs.output)})
        sid = full.session_id or full.id
        s = sessions.setdefault(sid, {"trace_id": full.id, "thread_id": sid, "user_input": _short(full.input),
                                      "steps": [], "final_answer": ""})
        s["steps"].append({"kind": "run", "input": full.input, "events": events})
        s["final_answer"] = _short(full.output)
    return list(sessions.values())[-limit:]


def _short(value, n: int = 600):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:n]


def format_trace(trace: dict, max_chars: int = 9000) -> str:
    lines = [f"USER REQUEST: {trace['user_input']}"]
    for step in trace["steps"]:
        lines.append(f"--- run ({step['kind']}): {step['input']}")
        for e in step["events"]:
            etype = e.get("type")
            if etype == "llm_tool_request":
                lines.append(f"AGENT CALLS {e['params']['name']}({json.dumps(e['params']['arguments'])})")
            elif etype == "tool_response":
                resp = e["response"]
                if "error" in resp:
                    err = resp["error"]
                    lines.append(f"  -> ERROR {err['code']} [{err.get('data', {}).get('stage')}]: {err['message']} "
                                 f"hint={err.get('data', {}).get('hint', '')}")
                else:
                    lines.append(f"  -> OK {_short(resp.get('result'), 300)}")
            elif etype == "hitl_triggered":
                lines.append(f"HITL TRIGGERED for {e['review'].get('claim_id')} ({e['review'].get('trigger')})")
            elif etype == "human_decision":
                lines.append(f"HUMAN DECISION {e['claim_id']}: {e['decision']} — {e.get('comment')}")
            elif etype == "final_answer":
                lines.append(f"AGENT ANSWER: {e.get('content', '')[:800]}")
            elif etype == "compaction":
                lines.append(f"CONTEXT COMPACTED ({e['removed_messages']} messages summarized)")
            elif "name" in e:  # Langfuse observation
                lines.append(f"[{e['type']}] {e['name']} level={e.get('level')} in={_short(e.get('input'), 300)} "
                             f"out={_short(e.get('output'), 300)}")
    lines.append(f"FINAL ANSWER: {trace['final_answer'][:1200]}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("judge did not return JSON")
    data = json.loads(match.group(0))
    data["score"] = max(1, min(5, int(data.get("score", 1))))
    return data


async def judge_all(traces: list[dict]) -> list[dict]:
    judge = create_chat_model("judge")
    results = []
    for i, trace in enumerate(traces, 1):
        prompt = JUDGE_PROMPT.format(trace=format_trace(trace))
        try:
            reply = await judge.ainvoke([HumanMessage(content=prompt)])
            verdict = parse_json(message_text(reply))
        except Exception as exc:  # noqa: BLE001
            verdict = {"score": None, "reasoning": f"judge failed: {exc}"}
        verdict.update(trace_id=trace["trace_id"], thread_id=trace["thread_id"], request=trace["user_input"])
        results.append(verdict)
        print(f"[{i}/{len(traces)}] score={verdict.get('score')}  {trace['user_input'][:70]}")
        await asyncio.sleep(1)
    return results


def write_report(results: list[dict], source: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    scores = [r["score"] for r in results if isinstance(r.get("score"), int)]
    avg = round(sum(scores) / len(scores), 2) if scores else None
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "source": source,
              "judge_provider": settings.llm_provider, "traces_evaluated": len(results),
              "average_error_recovery_score": avg, "results": results}
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# Error Recovery Evaluation (LLM-as-a-judge)", "",
          f"- Generated: {report['generated_at']}  |  Source: {source}  |  Judge: {settings.llm_provider}",
          f"- Traces evaluated: {len(results)}  |  **Average score: {avg} / 5**", "",
          "| # | Request | Score | Errors | Recovered | Reasoning |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        md.append(f"| {i} | {str(r['request'])[:60]} | {r.get('score')} | {r.get('errors_encountered', '-')} | "
                  f"{r.get('recovered', '-')} | {str(r.get('reasoning', '')).replace('|', '/')[:220]} |")
    (RESULTS_DIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nAverage error-recovery score: {avg} / 5  →  eval/results/report.md")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "local", "langfuse"], default="auto")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    source = args.source
    if source == "auto":
        source = "langfuse" if os.getenv("LANGFUSE_PUBLIC_KEY") else "local"
    traces = []
    if source == "langfuse":
        try:
            traces = load_langfuse_traces(args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load Langfuse traces ({exc}); falling back to local traces.")
            source = "local"
    if source == "local":
        traces = load_local_traces(args.limit)
    if not traces:
        raise SystemExit("No traces found. Run `python -m eval.run_scenarios` first.")
    print(f"Judging {len(traces)} traces from {source}...\n")
    write_report(await judge_all(traces), source)


if __name__ == "__main__":
    asyncio.run(main())
