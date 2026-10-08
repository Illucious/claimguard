"""TASK 3 — Tool provisioning.

Tools are just Python functions. Their type hints + docstrings are turned into
JSON schemas, either by LangChain (TOOL_MODE=local) or by the MCP server
(TOOL_MODE=mcp, Task 4). Every tool returns a JSON-serialisable dict.
Business errors are returned as {"error": {...}} instead of raising, so the
agent receives a structured message it can reason about.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from . import data_store as db
from . import rules

VALID_STATUSES = ["Approved", "Rejected", "Under Review", "Pending"]


def _error(code: int, message: str, hint: str = "") -> dict:
    return {"error": {"code": code, "message": message, "hint": hint}}


def _policy_view(p: dict) -> dict:
    end = rules.to_date(p.get("end_date"))
    return {
        "policy_id": p["policy_id"],
        "customer_name": p["customer_name"],
        "customer_phone": p.get("customer_phone", ""),
        "customer_email": p.get("customer_email", ""),
        "policy_type": p["policy_type"],
        "premium_annual": rules.to_float(p["premium_annual"]),
        "coverage_amount": rules.to_float(p["coverage_amount"]),
        "deductible": rules.to_float(p["deductible"]),
        "policy_status": p["policy_status"],
        "start_date": p["start_date"],
        "end_date": p["end_date"],
        "days_to_expiry": (end - date.today()).days if end else None,
        "covered_perils": rules.covered_perils(p),
    }


def _claim_view(c: dict) -> dict:
    return {
        "claim_id": c["claim_id"],
        "policy_id": c["policy_id"],
        "claim_type": c["claim_type"],
        "claim_amount": rules.to_float(c["claim_amount"]),
        "incident_date": c["incident_date"],
        "reported_date": c["reported_date"],
        "claim_status": c["claim_status"],
        "description": c["description"],
        "decision_note": c.get("decision_note", ""),
    }


def _load_claim_and_policy(claim_id: str):
    claim = db.get_claim(claim_id)
    if not claim:
        return None, None, _error(404, f"Claim {claim_id} was not found.", "Ask the user to confirm the claim ID.")
    policy = db.get_policy(claim["policy_id"])
    if not policy:
        return claim, None, _error(404, f"Policy {claim['policy_id']} linked to claim {claim_id} was not found.")
    return claim, policy, None


# ============================================================== information tools
def search_policies_by_customer(customer_name: str) -> dict:
    """Find policies belonging to a customer by (partial) name. Use when the user says "my policy" and gives a name but no policy ID.

    Args:
        customer_name: Full or partial customer name, e.g. "Amit Patel".
    """
    matches = db.find_policies_by_customer(customer_name)
    if not matches:
        return _error(404, f"No policies found for customer '{customer_name}'.", "Check the spelling or ask for the policy ID.")
    return {"count": len(matches), "matches": [_policy_view(p) for p in matches]}


def get_policy_details(policy_id: str) -> dict:
    """Get full details of an insurance policy: customer, type, annual premium, coverage, deductible, status, start/end dates and covered perils.

    Args:
        policy_id: Policy ID in the format P followed by 4 digits, e.g. "P1003".
    """
    policy = db.get_policy(policy_id)
    if not policy:
        return _error(404, f"Policy {policy_id} was not found.", "Ask the user to confirm the policy ID.")
    return _policy_view(policy)


def get_claim_details(claim_id: str) -> dict:
    """Get details of a single claim: linked policy, claim type, amount, incident/reported dates, current status and description.

    Args:
        claim_id: Claim ID in the format C followed by 4 digits, e.g. "C1003".
    """
    claim = db.get_claim(claim_id)
    if not claim:
        return _error(404, f"Claim {claim_id} was not found.", "Ask the user to confirm the claim ID.")
    return _claim_view(claim)


def list_claims_for_policy(policy_id: str) -> dict:
    """List all current claims filed under a policy.

    Args:
        policy_id: Policy ID in the format P followed by 4 digits, e.g. "P1003".
    """
    if not db.get_policy(policy_id):
        return _error(404, f"Policy {policy_id} was not found.")
    claims = db.claims_for_policy(policy_id)
    return {"policy_id": policy_id, "count": len(claims), "claims": [_claim_view(c) for c in claims]}


def get_claim_history(policy_id: str) -> dict:
    """Get the PREVIOUS (historical) claims of a policy, used for risk analysis: number of past claims and total amount paid.

    Args:
        policy_id: Policy ID in the format P followed by 4 digits, e.g. "P1003".
    """
    if not db.get_policy(policy_id):
        return _error(404, f"Policy {policy_id} was not found.")
    hist = db.history_for_policy(policy_id)
    paid = sum(rules.to_float(h["claim_amount"]) for h in hist if h["claim_status"].lower() == "paid")
    return {
        "policy_id": policy_id,
        "previous_claims_count": len(hist),
        "total_previous_paid": paid,
        "previous_claims": [{**h, "claim_amount": rules.to_float(h["claim_amount"])} for h in hist],
    }


# ================================================================ analysis tools
def check_claim_eligibility(claim_id: str) -> dict:
    """Check whether a claim is eligible: policy active, incident inside policy period, claim type covered, amount valid, within coverage.

    Args:
        claim_id: Claim ID in the format C followed by 4 digits, e.g. "C1003".
    """
    claim, policy, err = _load_claim_and_policy(claim_id)
    if err:
        return err
    return rules.check_eligibility(claim, policy)


def calculate_estimated_payout(claim_id: str, claim_amount: Optional[float] = None) -> dict:
    """Calculate the estimated payout of a claim: min(claim amount, coverage) minus deductible, or 0 if not eligible.

    Args:
        claim_id: Claim ID in the format C followed by 4 digits, e.g. "C1003".
        claim_amount: Optional what-if amount in rupees (plain number, no commas). Omit to use the claim's actual amount.
    """
    claim, policy, err = _load_claim_and_policy(claim_id)
    if err:
        return err
    eligibility = rules.check_eligibility(claim, policy)
    return rules.calculate_payout(claim, policy, eligibility, claim_amount)


def assess_claim_risk(claim_id: str) -> dict:
    """Run the full risk assessment of a claim using claim, policy and claim history. Returns risk score (0-100), level (LOW/MEDIUM/HIGH), risk factors, eligibility, estimated payout and a recommendation.

    Args:
        claim_id: Claim ID in the format C followed by 4 digits, e.g. "C1003".
    """
    claim, policy, err = _load_claim_and_policy(claim_id)
    if err:
        return err
    history = db.history_for_policy(policy["policy_id"])
    eligibility = rules.check_eligibility(claim, policy)
    payout = rules.calculate_payout(claim, policy, eligibility)
    return rules.assess_risk(claim, policy, history, eligibility, payout)


# =================================================================== action tool
def update_claim_status(claim_id: str, new_status: str, reason: str) -> dict:
    """Change the status of a claim (a real write action, guarded by hooks and human review). Allowed statuses: Approved, Rejected, Under Review, Pending.

    Args:
        claim_id: Claim ID in the format C followed by 4 digits, e.g. "C1003".
        new_status: One of "Approved", "Rejected", "Under Review", "Pending".
        reason: Short justification that will be stored with the claim.
    """
    if new_status not in VALID_STATUSES:
        return _error(400, f"Invalid status '{new_status}'.", f"Use one of {VALID_STATUSES}.")
    result = db.update_claim(claim_id, {
        "claim_status": new_status,
        "decision_note": reason,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    if not result:
        return _error(404, f"Claim {claim_id} was not found.")
    old, new = result
    return {"claim_id": claim_id, "previous_status": old["claim_status"], "new_status": new["claim_status"],
            "reason": reason, "updated_at": new["updated_at"]}


ALL_TOOLS = [
    search_policies_by_customer,
    get_policy_details,
    get_claim_details,
    list_claims_for_policy,
    get_claim_history,
    check_claim_eligibility,
    calculate_estimated_payout,
    assess_claim_risk,
    update_claim_status,
]
