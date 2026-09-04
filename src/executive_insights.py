from __future__ import annotations

import json
from typing import Any, Dict

DEFAULT_MODEL = "qwen3:4b"
MAX_OUTPUT_TOKENS = 500

EXECUTIVE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "portfolio_status": {
            "type": "string",
            "enum": ["Stable", "Watch", "Elevated", "Critical"],
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "priority_focus": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "watchouts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
    },
    "required": [
        "executive_summary",
        "portfolio_status",
        "key_findings",
        "priority_focus",
        "recommended_actions",
        "watchouts",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
You are the Executive AI layer of Retain-AI, an enterprise customer-retention
intelligence platform.

Your role is communication and synthesis, NOT prediction.

Authority hierarchy:
1. Calibrated XGBoost provides churn probability.
2. The Decision Engine provides risk tiers, Revenue at Risk, expected value,
   capacity selection and recommended interventions.
3. Historical portfolio trends describe model inference over time.
4. You summarize these signals for a senior business executive.

Rules:
- Never invent numbers, customers, causes, complaints, or trends.
- Use only the supplied context.
- Do not claim that correlation or SHAP proves causation.
- Do not say an intervention will save revenue; call it expected value under
  the stated assumptions.
- Revenue at Risk means P(churn) × annual contract value.
- Expected Save Value uses the explicit assumed 30% intervention success rate.
- Intervention cost is the explicit assumed ₹25,000 per account.
- Keep all monetary references in INR. Never use USD or '$'.
- Do not confuse churn probability with financial exposure.
- Do not recommend prioritising customers by probability alone.
- Explain concentration using both customer share and Revenue-at-Risk share.
- If usage/support signals do not indicate a problem, do not imply that they do.
- Treat the Decision Engine's recommended action as authoritative.
- Make the output useful to a CEO, CRO, CCO or business-unit leader: concise,
  commercially grounded and action-oriented.
- Do not reproduce the input context verbatim.
- portfolio_status should reflect the overall risk picture supplied in context.
"""


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _fmt_inr(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"₹{value:,.0f}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_cr(value: Any) -> str:
    try:
        return f"₹{float(value) / 1_00_00_000:.2f} Cr"
    except (TypeError, ValueError):
        return "N/A"


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    return {str(k): _clean(v) for k, v in dict(row).items()}


def prepare_executive_context(
    overview_decisions: Any,
    selected: Any,
    total_customers: int,
    total_acv: float,
    historical_predictions: Any,
    monthly_risk: Any,
    segment_concentration: Any,
    high_critical_count: int,
    overview_expected_save: float,
    overview_cost: float,
    overview_net_value: float,
    overview_benefit_cost: float,
    intervention_success_rate: float = 0.30,
    intervention_cost: float = 25_000,
    intervention_capacity: float = 0.10,
) -> Dict[str, Any]:
    """Build a compact, deterministic portfolio context for Executive AI."""

    risk_total = float(overview_decisions["revenue_at_risk"].sum())
    selected_risk = float(selected["revenue_at_risk"].sum())
    risk_coverage = selected_risk / risk_total if risk_total > 0 else 0.0

    risk_counts = (
        overview_decisions["risk_tier"]
        .value_counts()
        .to_dict()
    )

    # Top accounts are included only as business context; the LLM does not rank them.
    top_accounts = (
        selected.sort_values("priority_rank")
        .head(5)[
            [
                "customer_id",
                "segment",
                "risk_tier",
                "churn_probability",
                "annual_contract_value",
                "revenue_at_risk",
                "days_to_renewal",
                "behaviour",
                "recommended_action",
            ]
        ]
        .copy()
    )
    top_accounts_records = []
    for _, row in top_accounts.iterrows():
        top_accounts_records.append(
            {
                "customer_id": str(row.get("customer_id")),
                "segment": row.get("segment"),
                "risk_tier": row.get("risk_tier"),
                "churn_probability": _fmt_pct(row.get("churn_probability")),
                "annual_contract_value": _fmt_inr(row.get("annual_contract_value")),
                "revenue_at_risk": _fmt_inr(row.get("revenue_at_risk")),
                "days_to_renewal": _clean(row.get("days_to_renewal")),
                "behaviour": row.get("behaviour"),
                "recommended_action": row.get("recommended_action"),
            }
        )

    segments = []
    for _, row in segment_concentration.iterrows():
        segments.append(
            {
                "segment": row.get("segment"),
                "customers": int(row.get("customers", 0)),
                "customer_share": _fmt_pct(row.get("customer_share")),
                "revenue_at_risk": _fmt_inr(row.get("revenue_at_risk")),
                "risk_share": _fmt_pct(row.get("risk_share")),
                "risk_concentration": f"{float(row.get('risk_concentration_index', 0)):.2f}×",
            }
        )

    trend_start = historical_predictions["snapshot_date"].min()
    trend_end = historical_predictions["snapshot_date"].max()
    first_risk = float(monthly_risk.iloc[0, 0]) if len(monthly_risk) else 0.0
    latest_risk = float(monthly_risk.iloc[-1, 0]) if len(monthly_risk) else 0.0

    # Deterministic portfolio status signal. The LLM communicates it; it does not invent it.
    critical_share = risk_counts.get("Critical", 0) / total_customers if total_customers else 0
    high_critical_share = high_critical_count / total_customers if total_customers else 0
    if critical_share >= 0.05 or high_critical_share >= 0.15:
        status_signal = "Critical"
    elif high_critical_share >= 0.08 or latest_risk - first_risk >= 0.01:
        status_signal = "Elevated"
    elif high_critical_share >= 0.04 or latest_risk - first_risk >= 0.005:
        status_signal = "Watch"
    else:
        status_signal = "Stable"

    return {
        "portfolio": {
            "customers": int(total_customers),
            "total_acv": _fmt_inr(total_acv),
            "revenue_at_risk": _fmt_inr(risk_total),
            "revenue_at_risk_as_percent_of_acv": _fmt_pct(risk_total / total_acv if total_acv else 0),
            "average_predicted_churn_risk": _fmt_pct(overview_decisions["churn_probability"].mean()),
            "high_critical_customers": int(high_critical_count),
            "high_critical_share": _fmt_pct(high_critical_share),
            "risk_tier_counts": {str(k): int(v) for k, v in risk_counts.items()},
            "status_signal": status_signal,
        },
        "intervention": {
            "capacity": _fmt_pct(intervention_capacity),
            "accounts_selected": int(len(selected)),
            "revenue_at_risk_covered": _fmt_inr(selected_risk),
            "risk_coverage": _fmt_pct(risk_coverage),
            "expected_save_value": _fmt_inr(overview_expected_save),
            "intervention_cost": _fmt_inr(overview_cost),
            "net_expected_value": _fmt_inr(overview_net_value),
            "expected_benefit_cost": f"{overview_benefit_cost:.1f}×",
            "assumed_success_rate": _fmt_pct(intervention_success_rate),
            "assumed_cost_per_account": _fmt_inr(intervention_cost),
        },
        "trend": {
            "window": f"{trend_start.strftime('%d %b %Y')} to {trend_end.strftime('%d %b %Y')}",
            "first_month_average_risk": _fmt_pct(first_risk),
            "latest_month_average_risk": _fmt_pct(latest_risk),
            "change_percentage_points": f"{(latest_risk - first_risk) * 100:+.2f} pp",
        },
        "segments": segments,
        "top_priority_accounts": top_accounts_records,
    }


def _build_prompt(context: Dict[str, Any]) -> str:
    return f"""
Create an executive retention brief from the supplied Retain-AI portfolio context.

Return exactly the requested JSON schema.

The brief must answer:
1. What is the current portfolio situation?
2. Where is economic risk concentrated?
3. What should leadership focus on now?
4. What should leadership watch?

Important:
- Use the deterministic status_signal as the basis for portfolio_status.
- Use the segment data to describe concentration; do not invent causes.
- Use the top accounts only to illustrate exposure and action context.
- Recommended actions must be consistent with the supplied Decision Engine actions.
- Mention the 30% success rate and ₹25,000 cost only as assumptions when discussing economics.

PORTFOLIO CONTEXT:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""


def _validate(result: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "executive_summary",
        "portfolio_status",
        "key_findings",
        "priority_focus",
        "recommended_actions",
        "watchouts",
    ]
    for key in required:
        if key not in result:
            raise ValueError(f"Executive AI response missing field: {key}")

    if result["portfolio_status"] not in {"Stable", "Watch", "Elevated", "Critical"}:
        raise ValueError("Invalid executive portfolio status.")

    for key, size in {
        "key_findings": 3,
        "priority_focus": 3,
        "recommended_actions": 3,
        "watchouts": 2,
    }.items():
        if not isinstance(result[key], list) or len(result[key]) != size:
            raise ValueError(f"{key} must contain exactly {size} items.")
        if not all(isinstance(item, str) and item.strip() for item in result[key]):
            raise ValueError(f"{key} contains an invalid item.")

    return result


def generate_executive_insight(context: Dict[str, Any], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    from ollama import chat

    prompt = _build_prompt(context)

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + "\n\n/no_think"},
        ],
        format=EXECUTIVE_OUTPUT_SCHEMA,
        options={
            "temperature": 0.0,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
        think=False,
    )

    content = response["message"]["content"]
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Executive AI returned invalid JSON.") from exc

    return _validate(result)