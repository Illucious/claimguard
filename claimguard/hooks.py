"""TASK 5 — Lifecycle hooks: deterministic guardrails around a non-deterministic model.

pre_tool_hook  (runs BEFORE the tool is called)
    * tool exists, arguments match the discovered JSON schema (required / unknown / types)
    * business formats: policy_id P####, claim_id C####, claim_amount > 0 and sane
    * authorization for the write action update_claim_status
      (needs a risk assessment first; HIGH-risk / high-value claims need a human decision)
    -> on failure returns a simulated HTTP 400 (or 403) as a JSON-RPC error object,
       which goes back to the LLM so it can self-correct. The tool is never called.

post_tool_hook (runs AFTER the tool returns)
    * output is a dict, tool-level errors (404...) are normalised to JSON-RPC errors
    * business invariants: payout <= min(claim, coverage), payout 0 if ineligible,
      risk score 0-100 and consistent with level, HIGH risk must require review, etc.
      -> violation returns 422 so bad data is never presented as fact
    * PII masking (phone / email)
    * extracts the risk record and raises a human-review request (feeds Task 6)
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import settings

ID_RULES = {
    "policy_id": (re.compile(r"^P\d{4}$"), "P1003"),
    "claim_id": (re.compile(r"^C\d{4}$"), "C1003"),
}
VALID_STATUSES = ["Approved", "Rejected", "Under Review", "Pending"]
MAX_CLAIM_AMOUNT = 100_000_000  # ₹10 crore sanity cap
TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}


@dataclass
class HookOutcome:
    ok: bool
    args: dict = field(default_factory=dict)
    response: Optional[dict] = None          # JSON-RPC response (result or error)
    risk_record: Optional[dict] = None
    review_request: Optional[dict] = None


def jsonrpc_error(call_id: str, code: int, message: str, *, error_type: str, stage: str,
                  field_name: str | None = None, hint: str = "", extra: dict | None = None) -> dict:
    data = {"type": error_type, "stage": stage, "hint": hint}
    if field_name:
        data["field"] = field_name
    if extra:
        data.update(extra)
    return {"jsonrpc": "2.0", "id": call_id, "error": {"code": code, "message": message, "data": data}}


def _schema_type(prop: dict) -> Optional[str]:
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t is None and isinstance(prop.get("anyOf"), list):
        t = next((s.get("type") for s in prop["anyOf"] if s.get("type") != "null"), None)
    return t


def review_from_risk(record: dict, trigger: str) -> dict:
    return {
        "claim_id": record["claim_id"], "policy_id": record.get("policy_id"),
        "claim_type": record.get("claim_type"), "claim_amount": record.get("claim_amount"),
        "estimated_payout": record.get("estimated_payout"), "risk_level": record.get("risk_level"),
        "risk_score": record.get("risk_score"), "eligible": record.get("eligible"),
        "recommendation": record.get("recommendation"), "risk_factors": record.get("risk_factors_text", []),
        "trigger": trigger,
    }


# ============================================================== PRE-TOOL HOOK
def pre_tool_hook(call_id: str, name: str, args: Any, registry: dict,
                  risk_assessments: dict | None = None, human_decisions: dict | None = None) -> HookOutcome:
    def fail(code: int, msg: str, etype: str, field_name: str | None = None, hint: str = "", review=None):
        return HookOutcome(ok=False, response=jsonrpc_error(call_id, code, msg, error_type=etype, stage="pre_tool_hook",
                                                            field_name=field_name, hint=hint), review_request=review)

    if name not in registry:
        return fail(400, f"Unknown tool '{name}'.", "UnknownTool", hint=f"Available tools: {sorted(registry)}")
    if not isinstance(args, dict):
        return fail(400, "Tool arguments must be a JSON object.", "InvalidArguments")

    schema = registry[name].input_schema or {}
    props: dict = schema.get("properties", {})
    required: list = schema.get("required", [])

    clean: dict = {}
    for key, value in args.items():
        if key not in props:
            return fail(400, f"Unexpected argument '{key}' for {name}.", "UnexpectedField", key,
                        hint=f"Allowed arguments: {sorted(props)}")
        clean[key] = value.strip() if isinstance(value, str) else value

    for key in required:
        if clean.get(key) in (None, ""):
            return fail(400, f"Missing required argument '{key}' for {name}.", "MissingField", key,
                        hint=f"Provide '{key}'. If you don't know it, ask the user.")

    for key, value in clean.items():
        expected = _schema_type(props.get(key, {}))
        if value is None or expected not in TYPE_MAP:
            continue
        wrong = not isinstance(value, TYPE_MAP[expected]) or (expected in ("number", "integer") and isinstance(value, bool))
        if wrong:
            hint = "Send a plain number without commas or currency symbols, e.g. 300000." if expected == "number" else ""
            return fail(400, f"Argument '{key}' must be of type {expected}, got {type(value).__name__}.",
                        "InvalidType", key, hint=hint)

    for key, (pattern, example) in ID_RULES.items():
        if key in clean and clean[key] is not None and not pattern.match(str(clean[key])):
            return fail(400, f"Invalid {key} '{clean[key]}'.", "InvalidFormat", key,
                        hint=f"{key} must be an uppercase letter followed by 4 digits, e.g. {example}.")

    amount = clean.get("claim_amount")
    if amount is not None:
        if amount <= 0:
            return fail(400, f"claim_amount must be greater than 0, got {amount}.", "InvalidValue", "claim_amount",
                        hint="Ask the user for a positive amount.")
        if amount > MAX_CLAIM_AMOUNT:
            return fail(400, f"claim_amount {amount} exceeds the sanity limit of {MAX_CLAIM_AMOUNT}.", "InvalidValue",
                        "claim_amount", hint="Check the number of zeros.")

    if "customer_name" in clean and len(str(clean["customer_name"])) < 2:
        return fail(400, "customer_name is too short.", "InvalidValue", "customer_name")

    if name == "update_claim_status":
        return _authorize_status_update(call_id, clean, risk_assessments or {}, human_decisions or {}, fail)

    return HookOutcome(ok=True, args=clean)


def _authorize_status_update(call_id, clean, risk, decisions, fail) -> HookOutcome:
    canonical = {s.lower(): s for s in VALID_STATUSES}
    status = canonical.get(str(clean["new_status"]).lower())
    if not status:
        return fail(400, f"Invalid new_status '{clean['new_status']}'.", "InvalidValue", "new_status",
                    hint=f"Use one of {VALID_STATUSES}.")
    clean["new_status"] = status
    if len(str(clean.get("reason", ""))) < 10:
        return fail(400, "reason must be a meaningful sentence (at least 10 characters).", "InvalidValue", "reason")

    cid = clean["claim_id"]
    if status in ("Under Review", "Pending"):
        return HookOutcome(ok=True, args=clean)

    record = risk.get(cid)
    if not record:
        return fail(400, f"Cannot set {cid} to {status} before it has been assessed.", "PrerequisiteMissing",
                    "claim_id", hint=f"Call assess_claim_risk for {cid} first.")

    decision = decisions.get(cid)
    if decision:   # a human already decided → the agent may only record that exact decision
        expected = "Approved" if decision["decision"] == "approve" else "Rejected"
        if status != expected:
            return fail(403, f"Status {status} conflicts with the human reviewer's decision ({decision['decision']}).",
                        "ConflictsWithHumanDecision", "new_status", hint=f"Use new_status='{expected}'.")
        return HookOutcome(ok=True, args=clean)

    if status == "Approved":
        if record["eligible"] and record["risk_level"] == "LOW" and not record["requires_human_review"]:
            return HookOutcome(ok=True, args=clean)
        return fail(403, f"The agent is not allowed to approve {cid} (risk {record['risk_level']}, "
                         f"requires review: {record['requires_human_review']}). Human approval is required.",
                    "HumanApprovalRequired", "new_status",
                    hint="Do not retry. The claim has been sent to a human reviewer.",
                    review=review_from_risk(record, "Agent attempted to approve a claim that needs human approval"))

    # status == "Rejected"
    if not record["eligible"]:
        return HookOutcome(ok=True, args=clean)
    return fail(403, f"The agent is not allowed to reject eligible claim {cid}. Human decision required.",
                "HumanApprovalRequired", "new_status", hint="Do not retry. The claim has been sent to a human reviewer.",
                review=review_from_risk(record, "Agent attempted to reject an eligible claim"))


# ============================================================= POST-TOOL HOOK
def _mask_phone(v: str) -> str:
    digits = re.sub(r"\D", "", v or "")
    return ("*" * max(len(digits) - 4, 0) + digits[-4:]) if digits else v


def _mask_email(v: str) -> str:
    if not v or "@" not in v:
        return v
    user, domain = v.split("@", 1)
    return f"{user[:1]}***@{domain}"


def _mask_pii(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "customer_phone" and isinstance(v, str):
                out[k] = _mask_phone(v)
            elif k == "customer_email" and isinstance(v, str):
                out[k] = _mask_email(v)
            else:
                out[k] = _mask_pii(v)
        return out
    if isinstance(node, list):
        return [_mask_pii(v) for v in node]
    return node


def _expected_level(score: float) -> str:
    if score >= settings.risk_high_min:
        return "HIGH"
    if score >= settings.risk_medium_min:
        return "MEDIUM"
    return "LOW"


def _check_policy(p: dict) -> list[str]:
    v = []
    for key in ("policy_id", "premium_annual", "coverage_amount", "policy_status", "start_date", "end_date"):
        if key not in p:
            v.append(f"missing field '{key}'")
    if p.get("premium_annual", 1) <= 0:
        v.append("premium must be > 0")
    if p.get("coverage_amount", 1) <= 0:
        v.append("coverage must be > 0")
    if p.get("deductible", 0) < 0 or p.get("deductible", 0) >= p.get("coverage_amount", float("inf")):
        v.append("deductible must be >= 0 and < coverage")
    if str(p.get("start_date", "")) >= str(p.get("end_date", "~")):
        v.append("start_date must be before end_date")
    return v


def _validate(name: str, r: dict, args: dict) -> list[str]:
    v: list[str] = []
    if name == "get_policy_details":
        v += _check_policy(r)
        if r.get("policy_id") != args.get("policy_id"):
            v.append("returned policy_id does not match the requested one")
    elif name == "search_policies_by_customer":
        for p in r.get("matches", []):
            v += _check_policy(p)
    elif name == "get_claim_details":
        if r.get("claim_id") != args.get("claim_id"):
            v.append("returned claim_id does not match the requested one")
        if r.get("claim_amount", 0) <= 0:
            v.append("claim_amount must be > 0")
    elif name == "check_claim_eligibility":
        if not isinstance(r.get("eligible"), bool):
            v.append("'eligible' must be a boolean")
        if r.get("eligible") is False and not r.get("failed_reasons"):
            v.append("ineligible result must list failed_reasons")
    elif name == "calculate_estimated_payout":
        payout, amount, cov = r.get("estimated_payout", -1), r.get("claim_amount", 0), r.get("coverage_amount", 0)
        if payout < 0:
            v.append("payout cannot be negative")
        if payout > min(amount, cov) + 0.01:
            v.append(f"payout {payout} exceeds min(claim {amount}, coverage {cov})")
        if r.get("eligible") is False and payout != 0:
            v.append("ineligible claim must have payout 0")
    elif name == "assess_claim_risk":
        score, level = r.get("risk_score", -1), r.get("risk_level")
        if not 0 <= score <= 100:
            v.append(f"risk_score {score} outside 0-100")
        elif level != _expected_level(score):
            v.append(f"risk_level {level} inconsistent with score {score}")
        if r.get("eligible") and level == "HIGH" and not r.get("requires_human_review"):
            v.append("eligible HIGH-risk claim must require human review")
        if r.get("estimated_payout", 0) > r.get("claim_amount", 0) + 0.01:
            v.append("estimated payout exceeds claim amount")
    elif name == "update_claim_status":
        if r.get("new_status") != args.get("new_status"):
            v.append("status was not updated to the requested value")
    return v


def post_tool_hook(call_id: str, name: str, args: dict, raw: Any) -> HookOutcome:
    if not isinstance(raw, dict):
        return HookOutcome(ok=False, response=jsonrpc_error(call_id, 502, "Tool returned a non-JSON-object result.",
                                                            error_type="BadToolOutput", stage="post_tool_hook"))
    if "error" in raw:   # business error from the tool itself (e.g. 404) → pass through as JSON-RPC error
        err = raw["error"] if isinstance(raw["error"], dict) else {"message": str(raw["error"])}
        return HookOutcome(ok=False, response=jsonrpc_error(call_id, int(err.get("code", 500)), err.get("message", "Tool error"),
                                                            error_type="ToolError", stage="tool", hint=err.get("hint", "")))

    violations = _validate(name, raw, args)
    if violations:
        return HookOutcome(ok=False, response=jsonrpc_error(
            call_id, 422, "Tool output failed business-rule validation; do not present it as fact.",
            error_type="OutputValidationFailed", stage="post_tool_hook", extra={"violations": violations}))

    result = _mask_pii(copy.deepcopy(raw))
    outcome = HookOutcome(ok=True, args=args, response={"jsonrpc": "2.0", "id": call_id, "result": result})

    if name == "assess_claim_risk":
        record = {k: result.get(k) for k in ("claim_id", "policy_id", "claim_type", "claim_amount", "estimated_payout",
                                             "risk_level", "risk_score", "eligible", "requires_human_review",
                                             "high_value", "recommendation")}
        record["risk_factors_text"] = [f"{f['detail']} (+{f['points']})" for f in result.get("risk_factors", [])]
        outcome.risk_record = record
        if record["requires_human_review"]:
            trigger = "HIGH risk" if record["risk_level"] == "HIGH" else "High-value claim"
            if record["risk_level"] == "HIGH" and record["high_value"]:
                trigger = "HIGH risk and high-value claim"
            outcome.review_request = review_from_risk(record, trigger)
    return outcome
