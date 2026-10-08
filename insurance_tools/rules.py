"""Deterministic insurance business rules.

The LLM never calculates payouts or risk itself — it calls tools that run
these rules, so the numbers are always reproducible and auditable.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Optional

HIGH_VALUE_THRESHOLD = float(os.getenv("HIGH_VALUE_THRESHOLD", "500000"))
RISK_MEDIUM_MIN = int(os.getenv("RISK_MEDIUM_MIN", "30"))
RISK_HIGH_MIN = int(os.getenv("RISK_HIGH_MIN", "60"))
NEW_POLICY_WINDOW_DAYS = 30
LATE_REPORTING_DAYS = 30


# ------------------------------------------------------------------ helpers
def to_float(value, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def to_date(value) -> Optional[date]:
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return None


def covered_perils(policy: dict) -> list[str]:
    return [p.strip() for p in policy.get("covered_perils", "").split("|") if p.strip()]


def risk_level_for(score: int) -> str:
    if score >= RISK_HIGH_MIN:
        return "HIGH"
    if score >= RISK_MEDIUM_MIN:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------- eligibility
def check_eligibility(claim: dict, policy: dict) -> dict:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str, blocking: bool = True):
        checks.append({"check": name, "passed": passed, "blocking": blocking, "detail": detail})

    status = policy.get("policy_status", "")
    add("policy_active", status.lower() == "active", f"Policy status is '{status}'.")

    incident = to_date(claim.get("incident_date"))
    start, end = to_date(policy.get("start_date")), to_date(policy.get("end_date"))
    in_period = bool(incident and start and end and start <= incident <= end)
    add("incident_within_policy_period", in_period,
        f"Incident {claim.get('incident_date')} vs policy period {policy.get('start_date')} to {policy.get('end_date')}.")

    perils = covered_perils(policy)
    ctype = claim.get("claim_type", "")
    covered = ctype.lower() in {p.lower() for p in perils}
    add("claim_type_covered", covered, f"Claim type '{ctype}'; covered perils: {', '.join(perils) or 'none'}.")

    amount = to_float(claim.get("claim_amount"))
    add("claim_amount_positive", amount > 0, f"Claim amount is {amount:,.0f}.")

    coverage = to_float(policy.get("coverage_amount"))
    add("within_coverage_limit", amount <= coverage,
        f"Claim {amount:,.0f} vs coverage {coverage:,.0f}. Payout will be capped at coverage if exceeded.",
        blocking=False)

    reported = to_date(claim.get("reported_date"))
    add("reported_after_incident", bool(reported and incident and reported >= incident),
        f"Reported {claim.get('reported_date')}, incident {claim.get('incident_date')}.")

    failed = [c["detail"] for c in checks if c["blocking"] and not c["passed"]]
    warnings = [c["detail"] for c in checks if not c["blocking"] and not c["passed"]]
    return {
        "claim_id": claim.get("claim_id"),
        "policy_id": policy.get("policy_id"),
        "eligible": not failed,
        "checks": checks,
        "failed_reasons": failed,
        "warnings": warnings,
    }


# -------------------------------------------------------------------- payout
def calculate_payout(claim: dict, policy: dict, eligibility: dict,
                     claim_amount_override: Optional[float] = None) -> dict:
    amount = claim_amount_override if claim_amount_override is not None else to_float(claim.get("claim_amount"))
    coverage = to_float(policy.get("coverage_amount"))
    deductible = to_float(policy.get("deductible"))
    base = {
        "claim_id": claim.get("claim_id"),
        "policy_id": policy.get("policy_id"),
        "claim_amount": amount,
        "coverage_amount": coverage,
        "deductible": deductible,
        "is_what_if": claim_amount_override is not None,
        "eligible": eligibility["eligible"],
    }
    if not eligibility["eligible"]:
        return {**base, "estimated_payout": 0.0, "capped_by_coverage": False,
                "calculation": "Claim is not eligible, so the payout is 0.",
                "not_eligible_reasons": eligibility["failed_reasons"]}

    covered_amount = min(amount, coverage)
    payout = max(covered_amount - deductible, 0.0)
    return {**base, "estimated_payout": payout, "capped_by_coverage": amount > coverage,
            "calculation": f"min(claim {amount:,.0f}, coverage {coverage:,.0f}) - deductible {deductible:,.0f} = {payout:,.0f}"}


# ---------------------------------------------------------------------- risk
def assess_risk(claim: dict, policy: dict, history: list[dict],
                eligibility: dict, payout: dict) -> dict:
    factors: list[dict] = []

    def add(points: int, factor: str, detail: str):
        factors.append({"factor": factor, "points": points, "detail": detail})

    amount = to_float(claim.get("claim_amount"))
    coverage = to_float(policy.get("coverage_amount")) or 1.0
    prev_count = len(history)
    prev_paid = sum(to_float(h.get("claim_amount")) for h in history if h.get("claim_status", "").lower() == "paid")

    if prev_count >= 2:
        add(30, "frequent_claimant", f"{prev_count} previous claims on this policy.")
    elif prev_count == 1:
        add(10, "previous_claim", "1 previous claim on this policy.")

    if prev_paid / coverage > 0.5:
        add(20, "high_previous_payouts", f"Previous paid claims {prev_paid:,.0f} exceed 50% of coverage.")

    if amount / coverage > 0.5:
        add(20, "large_claim_vs_coverage", f"Claim is {amount / coverage:.0%} of coverage.")

    high_value = amount >= HIGH_VALUE_THRESHOLD
    if high_value:
        add(15, "high_value_claim", f"Claim {amount:,.0f} is at or above the high-value threshold {HIGH_VALUE_THRESHOLD:,.0f}.")

    incident, start = to_date(claim.get("incident_date")), to_date(policy.get("start_date"))
    if incident and start and 0 <= (incident - start).days <= NEW_POLICY_WINDOW_DAYS:
        add(20, "early_claim", f"Incident occurred {(incident - start).days} days after policy start.")

    if claim.get("claim_type", "").lower() == "theft":
        add(10, "theft_claim", "Theft claims carry higher fraud risk.")

    reported = to_date(claim.get("reported_date"))
    if incident and reported and (reported - incident).days > LATE_REPORTING_DAYS:
        add(10, "late_reporting", f"Reported {(reported - incident).days} days after the incident.")

    score = min(sum(f["points"] for f in factors), 100)
    level = risk_level_for(score)
    eligible = eligibility["eligible"]
    requires_review = eligible and (level == "HIGH" or high_value)

    if not eligible:
        recommendation = "Reject - claim is not eligible"
        reason = "; ".join(eligibility["failed_reasons"])
    elif requires_review:
        recommendation = "Manual Review Required"
        reason = (f"Risk level {level} (score {score})" + (" and high-value claim" if high_value else "")
                  + ". A human reviewer must approve or reject.")
    elif level == "MEDIUM":
        recommendation = "Proceed with standard verification"
        reason = f"Moderate risk (score {score}); verify documents before settlement."
    else:
        recommendation = "Fast-track approval recommended"
        reason = f"Low risk (score {score}), eligible and within coverage."

    return {
        "claim_id": claim.get("claim_id"),
        "policy_id": policy.get("policy_id"),
        "claim_type": claim.get("claim_type"),
        "claim_amount": amount,
        "risk_score": score,
        "risk_level": level,
        "risk_factors": factors,
        "high_value": high_value,
        "requires_human_review": requires_review,
        "eligible": eligible,
        "eligibility_failed_reasons": eligibility["failed_reasons"],
        "estimated_payout": payout["estimated_payout"],
        "previous_claims_count": prev_count,
        "total_previous_paid": prev_paid,
        "recommendation": recommendation,
        "recommendation_reason": reason,
    }
