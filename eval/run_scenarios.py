"""Generate historical traces for the judge by running 10 error-prone scenarios.

    python -m eval.run_scenarios

Each scenario runs on its own thread. HITL pauses are auto-answered with the
scenario's `auto_review` decision. Traces are sent to Langfuse (if configured)
and appended to data/traces/traces.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from claimguard.agent import ClaimGuardAgent

SCENARIOS = Path(__file__).with_name("scenarios.json")


async def main() -> None:
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    agent = await ClaimGuardAgent().start()
    run_tag = uuid.uuid4().hex[:6]
    try:
        for sc in scenarios:
            thread_id = f"eval-{run_tag}-{sc['id']}"
            print(f"\n=== {sc['id']}: {sc['message']}  ({sc['tests']})")
            result = await agent.chat(thread_id, sc["message"], user_id="eval-runner")
            while result["status"] == "awaiting_human_review":
                decision = sc.get("auto_review", "reject")
                print(f"   HITL pause on {result['review']['claim_id']} -> auto '{decision}'")
                result = await agent.resume(thread_id, decision, f"Auto-{decision} by eval runner", "eval-runner")
            print(f"   tools: {result['tools_used']}  errors: {result['errors_in_run']}")
            print(f"   answer: {result['answer'][:300]}")
            await asyncio.sleep(1)  # be gentle with free-tier rate limits
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
