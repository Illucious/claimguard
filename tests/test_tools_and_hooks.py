"""Offline tests: business rules (Task 3) and hooks (Task 5). No LLM / API keys needed."""
from insurance_tools import tools as T
from claimguard.hooks import post_tool_hook, pre_tool_hook
from claimguard.tool_provider import LocalToolProvider


async def _registry():
    p = LocalToolProvider()
    await p.start()
    return p.registry


def test_policy_lookup(temp_data):
    p = T.get_policy_details("P1003")
    assert p["premium_annual"] == 24500 and p["coverage_amount"] == 800000
    assert T.get_policy_details("P9999")["error"]["code"] == 404


def test_c1003_is_high_risk_and_needs_review(temp_data):
    r = T.assess_claim_risk("C1003")
    assert r["eligible"] and r["risk_level"] == "HIGH" and r["requires_human_review"]
    assert r["estimated_payout"] == 440000  # 4,50,000 - 10,000 deductible


def test_ineligible_and_capped_claims(temp_data):
    assert T.check_claim_eligibility("C1009")["eligible"] is False      # theft not covered
    assert T.check_claim_eligibility("C1005")["eligible"] is False      # expired policy
    assert T.check_claim_eligibility("C1011")["eligible"] is False      # incident before policy start
    payout = T.calculate_estimated_payout("C1012")
    assert payout["capped_by_coverage"] and payout["estimated_payout"] == 790000
    low = T.assess_claim_risk("C1001")
    assert low["risk_level"] == "LOW" and not low["requires_human_review"]


async def test_pre_hook_rejects_bad_ids_with_400(temp_data):
    reg = await _registry()
    out = pre_tool_hook("1", "get_claim_details", {"claim_id": "c1003"}, reg)
    assert not out.ok and out.response["error"]["code"] == 400
    assert "C1003" in out.response["error"]["data"]["hint"]
    assert pre_tool_hook("2", "get_claim_details", {}, reg).response["error"]["data"]["type"] == "MissingField"
    assert pre_tool_hook("3", "nope", {}, reg).response["error"]["data"]["type"] == "UnknownTool"
    neg = pre_tool_hook("4", "calculate_estimated_payout", {"claim_id": "C1001", "claim_amount": -5}, reg)
    assert neg.response["error"]["code"] == 400
    assert pre_tool_hook("5", "get_claim_details", {"claim_id": "C1003"}, reg).ok


async def test_pre_hook_blocks_unapproved_high_risk_approval(temp_data):
    reg = await _registry()
    args = {"claim_id": "C1003", "new_status": "Approved", "reason": "Looks fine to me"}
    no_assessment = pre_tool_hook("1", "update_claim_status", args, reg)
    assert no_assessment.response["error"]["data"]["type"] == "PrerequisiteMissing"

    risk = post_tool_hook("2", "assess_claim_risk", {"claim_id": "C1003"}, T.assess_claim_risk("C1003"))
    assert risk.review_request is not None
    blocked = pre_tool_hook("3", "update_claim_status", args, reg, {"C1003": risk.risk_record})
    assert blocked.response["error"]["code"] == 403 and blocked.review_request["claim_id"] == "C1003"

    approved = pre_tool_hook("4", "update_claim_status", args, reg, {"C1003": risk.risk_record},
                             {"C1003": {"decision": "approve"}})
    assert approved.ok


def test_post_hook_catches_bad_output_and_masks_pii(temp_data):
    bad = {"claim_id": "C1003", "claim_amount": 450000, "coverage_amount": 800000, "estimated_payout": 900000,
           "eligible": True}
    out = post_tool_hook("1", "calculate_estimated_payout", {"claim_id": "C1003"}, bad)
    assert out.response["error"]["code"] == 422
    good = post_tool_hook("2", "get_policy_details", {"policy_id": "P1003"}, T.get_policy_details("P1003"))
    assert good.response["result"]["customer_phone"].endswith("9898") and "*" in good.response["result"]["customer_phone"]
    nf = post_tool_hook("3", "get_claim_details", {"claim_id": "C9999"}, T.get_claim_details("C9999"))
    assert nf.response["error"]["code"] == 404
