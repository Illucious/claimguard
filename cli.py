"""Interactive terminal client.

    python cli.py                      # new conversation
    python cli.py --thread demo-1      # continue / resume a conversation (survives restarts)

Commands:  /state  show thread state   /tools  list discovered tools
           /new    start a new thread  /quit   exit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid

from claimguard.agent import ClaimGuardAgent, ThreadAwaitingReviewError


def inr(value) -> str:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return str(value)
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if n < 0 else "") + "₹" + s


def print_review_card(review: dict) -> None:
    print("\n" + "=" * 64)
    print("  ⚠  HUMAN APPROVAL REQUIRED")
    print("=" * 64)
    print(f"  Claim:            {review.get('claim_id')}  (policy {review.get('policy_id')})")
    print(f"  Type:             {review.get('claim_type')}")
    print(f"  Claim amount:     {inr(review.get('claim_amount'))}")
    print(f"  Estimated payout: {inr(review.get('estimated_payout'))}")
    print(f"  Risk:             {review.get('risk_level')} (score {review.get('risk_score')})")
    print(f"  Trigger:          {review.get('trigger')}")
    print(f"  AI recommends:    {review.get('recommendation')}")
    for factor in review.get("risk_factors") or []:
        print(f"    • {factor}")
    print("=" * 64)


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip()


async def handle_review_loop(agent: ClaimGuardAgent, thread_id: str, result: dict) -> dict:
    while result["status"] == "awaiting_human_review":
        print_review_card(result["review"])
        decision = ""
        while decision.lower() not in ("approve", "reject"):
            decision = await ask("  Type 'Approve' or 'Reject': ")
        comment = await ask("  Comment for the record (optional): ")
        result = await agent.resume(thread_id, decision, comment or f"{decision} via CLI", reviewer="cli-reviewer")
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="ClaimGuard AI terminal")
    parser.add_argument("--thread", default=None, help="thread id to continue")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    agent = await ClaimGuardAgent().start()
    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:6]}"
    print(f"\nClaimGuard AI  |  tools via {agent.provider.mode.upper()}: {len(agent.provider.tools)} discovered"
          f"  |  telemetry: {'Langfuse' if agent.telemetry.enabled else 'off'}")
    print(f"Thread: {thread_id}   (re-run with --thread {thread_id} to resume after a restart)")
    print("Try: 'What is the premium for P1003?'  |  'Assess claim C1003 and tell me if it should proceed'\n")

    try:
        # If we restarted while a review was pending, offer it immediately.
        pending = (await agent.get_thread(thread_id))["pending_review"]
        if pending:
            print("This thread was paused waiting for human review.")
            result = await handle_review_loop(agent, thread_id, {"status": "awaiting_human_review", "review": pending})
            print(f"\nClaimGuard > {result['answer']}\n")

        while True:
            message = await ask("You > ")
            if not message:
                continue
            if message in ("/quit", "/exit"):
                break
            if message == "/new":
                thread_id = f"cli-{uuid.uuid4().hex[:6]}"
                print(f"New thread: {thread_id}")
                continue
            if message == "/tools":
                for t in agent.provider.tools:
                    print(f"  - {t.name}: {t.description.splitlines()[0]}")
                continue
            if message == "/state":
                state = await agent.get_thread(thread_id)
                state.pop("messages")
                print(json.dumps(state, indent=2, ensure_ascii=False, default=str))
                continue

            try:
                result = await agent.chat(thread_id, message, user_id="cli-user")
            except ThreadAwaitingReviewError as exc:
                print(f"! {exc}")
                continue
            if result["tools_used"]:
                print(f"  [tools: {', '.join(result['tools_used'])} | errors self-corrected: {result['errors_in_run']}"
                      f"{' | context compacted' if result['compacted'] else ''}]")
            result = await handle_review_loop(agent, thread_id, result)
            print(f"\nClaimGuard > {result['answer']}\n")
            if result.get("trace_url"):
                print(f"  trace: {result['trace_url']}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
