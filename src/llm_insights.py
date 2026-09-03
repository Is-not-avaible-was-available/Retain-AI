from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ollama import chat


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "qwen3:4b"

# Keep local generation short.
# The LLM is a communication layer, not the analytics engine.
MAX_OUTPUT_TOKENS = 350


# ============================================================
# Structured LLM output schema
# ============================================================

LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_summary": {"type": "string"},
        "key_drivers": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "recommended_action": {"type": "string"},
        "reasoning": {"type": "string"},
        "priority": {
            "type": "string",
            "enum": ["Low", "Moderate", "High", "Critical"],
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": [
        "risk_summary",
        "key_drivers",
        "recommended_action",
        "reasoning",
        "priority",
        "next_steps",
    ],
    "additionalProperties": False,
}


# ============================================================
# Utility functions
# ============================================================

def _clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""

    if value is None:
        return None

    # Handle NaN / NaT.
    try:
        if value != value:
            return None
    except Exception:
        pass

    # Convert numpy scalar.
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    # Convert pandas Timestamp / datetime.
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


def _clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Make dictionary values JSON-safe."""

    return {
        str(key): _clean_value(value)
        for key, value in data.items()
    }


def _format_inr(value: Any) -> str:
    """Format a numeric monetary value as Indian Rupees."""

    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    return f"₹{numeric_value:,.0f}"


def _format_percentage(value: Any) -> str:
    """Format a probability such as 0.0831 as 8.31%."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _get_row(value: Any, name: str) -> Any:
    """
    Normalize a one-row DataFrame, Series, or dictionary into a row-like object.
    """

    if value is None:
        return None

    # One-row DataFrame.
    if hasattr(value, "iloc") and hasattr(value, "columns"):
        if len(value) != 1:
            raise ValueError(
                f"{name} must contain exactly one customer record."
            )
        return value.iloc[0]

    # Series / dictionary / other mapping-like object can be used directly.
    return value


def _row_get(row: Any, column: str, default: Any = None) -> Any:
    """Safely retrieve and clean a value from a row-like object."""

    if row is None:
        return default

    try:
        return _clean_value(row[column])
    except Exception:
        pass

    # Also support ordinary objects with attributes.
    try:
        return _clean_value(getattr(row, column))
    except Exception:
        return default


# ============================================================
# Customer context
# ============================================================

def prepare_customer_context(
    customer_row: Any,
    decision_row: Any = None,
    explanation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a customer feature row and its Decision Engine row into a compact,
    business-focused context for the local LLM.

    Parameters
    ----------
    customer_row:
        One-row customer feature DataFrame, Series, or dictionary. This is the
        source for customer profile, health, usage, support, and contract data.

    decision_row:
        One-row Decision Engine DataFrame, Series, or dictionary. This is the
        authoritative source for model risk, financial calculations, signals,
        recommended action, and prioritisation.

    explanation:
        SHAP explanation dictionary for the customer.

    The LLM receives interpreted business signals rather than the complete
    101-column feature vector.
    """

    row = _get_row(customer_row, "customer_row")
    decision = _get_row(decision_row, "decision_row")

    # ========================================================
    # Build context
    # ========================================================

    context = {

        # ----------------------------------------------------
        # Customer profile
        # ----------------------------------------------------

        "customer": {
            "customer_id": _row_get(row, "customer_id"),
            "industry": _row_get(row, "industry"),
            "region": _row_get(row, "region"),
            "segment": _row_get(row, "segment"),
            "company_size": _row_get(row, "company_size"),
            "acquisition_channel": _row_get(row, "acquisition_channel"),
        },

        # ----------------------------------------------------
        # Model risk -- Decision Engine is authoritative
        # ----------------------------------------------------

        "risk": {
            "churn_probability": _row_get(
                decision,
                "churn_probability",
                _row_get(row, "churn_probability"),
            ),
            "risk_tier": _row_get(
                decision,
                "risk_tier",
                _row_get(row, "risk_tier"),
            ),
            "exposure_tier": _row_get(
                decision,
                "exposure_tier",
                _row_get(row, "exposure_tier"),
            ),
            "priority_rank": _row_get(
                decision,
                "priority_rank",
                _row_get(row, "priority_rank"),
            ),
        },

        # ----------------------------------------------------
        # Financial exposure -- Decision Engine is authoritative
        # ----------------------------------------------------

        "financial": {
            "annual_contract_value": _row_get(
                decision,
                "annual_contract_value",
                _row_get(row, "annual_contract_value"),
            ),
            "total_contract_value": _row_get(
                decision,
                "total_contract_value",
                _row_get(row, "total_contract_value"),
            ),
            "revenue_at_risk": _row_get(
                decision,
                "revenue_at_risk",
                _row_get(row, "revenue_at_risk"),
            ),
            "expected_save_value": _row_get(
                decision,
                "expected_save_value",
                _row_get(row, "expected_save_value"),
            ),
            "intervention_cost": _row_get(
                decision,
                "intervention_cost",
                _row_get(row, "intervention_cost"),
            ),
            "net_expected_value": _row_get(
                decision,
                "net_expected_value",
                _row_get(row, "net_expected_value"),
            ),
            "expected_benefit_cost_ratio": _row_get(
                decision,
                "expected_benefit_cost_ratio",
                _row_get(row, "expected_benefit_cost_ratio"),
            ),
        },

        # ----------------------------------------------------
        # Customer health -- feature store
        # ----------------------------------------------------

        "health": {
            "health_score": _row_get(row, "health_score"),
            "baseline_health": _row_get(row, "baseline_health"),
            "health_trend": _row_get(row, "health_trend"),
            "behaviour": _row_get(row, "behaviour"),
        },

        # ----------------------------------------------------
        # Usage / adoption -- feature store
        # ----------------------------------------------------

        "usage": {
            "active_users": _row_get(row, "active_users"),
            "sessions": _row_get(row, "sessions"),
            "active_days": _row_get(row, "active_days"),
            "features_used": _row_get(row, "features_used"),
            "product_count": _row_get(row, "product_count"),
            "active_users_change_12w": _row_get(row, "active_users_change_12w"),
            "sessions_change_12w": _row_get(row, "sessions_change_12w"),
            "usage_minutes_change_12w": _row_get(
                row,
                "usage_minutes_change_12w",
            ),
            "usage_decline_12w_flag": _row_get(
                row,
                "usage_decline_12w_flag",
            ),
            "severe_usage_decline_12w_flag": _row_get(
                row,
                "severe_usage_decline_12w_flag",
            ),
        },

        # ----------------------------------------------------
        # Support / customer experience -- feature store
        # ----------------------------------------------------

        "support": {
            "tickets_12w": _row_get(row, "tickets_12w"),
            "escalations_12w": _row_get(row, "escalations_12w"),
            "reopens_12w": _row_get(row, "reopens_12w"),
            "critical_tickets_12w": _row_get(row, "critical_tickets_12w"),
            "high_priority_tickets_12w": _row_get(
                row,
                "high_priority_tickets_12w",
            ),
            "avg_csat_12w": _row_get(row, "avg_csat_12w"),
            "avg_resolution_hours_12w": _row_get(
                row,
                "avg_resolution_hours_12w",
            ),
            "high_support_burden_12w": _row_get(
                row,
                "high_support_burden_12w",
            ),
            "high_escalation_12w": _row_get(
                row,
                "high_escalation_12w",
            ),
            "low_csat_12w": _row_get(row, "low_csat_12w"),
            "slow_resolution_12w": _row_get(
                row,
                "slow_resolution_12w",
            ),
        },

        # ----------------------------------------------------
        # Contract / renewal
        # ----------------------------------------------------

        "renewal": {
            "contract_type": _row_get(row, "contract_type"),
            "auto_renew": _row_get(row, "auto_renew"),
            "days_to_renewal": _row_get(row, "days_to_renewal"),
            "renewal_within_30d": _row_get(row, "renewal_within_30d"),
            "renewal_within_90d": _row_get(row, "renewal_within_90d"),
            "renewal_within_180d": _row_get(row, "renewal_within_180d"),
            "renewal_urgency": _row_get(
                decision,
                "renewal_urgency",
                _row_get(row, "renewal_urgency"),
            ),
        },

        # ----------------------------------------------------
        # Decision engine -- authoritative business signals
        # ----------------------------------------------------

        "decision": {
            "health_signal": _row_get(decision, "health_signal"),
            "usage_signal": _row_get(decision, "usage_signal"),
            "support_signal": _row_get(decision, "support_signal"),
            "recommended_action": _row_get(
                decision,
                "recommended_action",
            ),
            "recommended_interventions": _row_get(
                decision,
                "recommended_interventions",
            ),
            "economically_viable": _row_get(
                decision,
                "economically_viable",
            ),
            "selected_for_intervention": _row_get(
                decision,
                "selected_for_intervention",
            ),
        },
    }

    # ========================================================
    # SHAP explanation
    # ========================================================

    if explanation:

        positive_drivers = explanation.get("positive_drivers", [])
        negative_drivers = explanation.get("negative_drivers", [])

        def simplify_driver(driver: Dict[str, Any]) -> Dict[str, Any]:
            display_name = (
                driver.get("display_name")
                or driver.get("feature")
            )

            return {
                "feature": driver.get("feature"),
                "display_name": display_name,
                "value": _clean_value(driver.get("value")),
                "shap_value": _clean_value(driver.get("shap_value")),
            }

        context["model_explanation"] = {
            "positive_drivers": [
                simplify_driver(driver)
                for driver in positive_drivers[:5]
            ],
            "negative_drivers": [
                simplify_driver(driver)
                for driver in negative_drivers[:5]
            ],
        }

    return context


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """
You are the natural-language explanation layer of a customer retention

decision system.

The analytics are already complete.

You MUST follow these authorities:

XGBoost:
- authoritative for churn_probability.

SHAP:
- authoritative for model drivers.
- positive SHAP means the feature increases the model's predicted risk.
- negative SHAP means the feature decreases the model's predicted risk.
- SHAP does not establish causality.

Decision Engine:
- authoritative for risk_tier.
- authoritative for recommended_action.
- authoritative for financial calculations.
- authoritative for exposure_tier and priority_rank.
- authoritative for health_signal, usage_signal, support_signal,
  renewal_urgency, and economic viability.

CRITICAL RULES:

1. Never calculate or modify churn_probability.

2. The output field "priority" MUST equal the supplied
   Decision Engine risk_tier.

3. The output field "recommended_action" MUST equal the supplied
   Decision Engine recommended_action.

4. Do not confuse risk_tier with exposure_tier.

5. High financial exposure does not automatically mean high churn risk.

6. All monetary values are INR. Use ₹. Never use $ or USD.

7. Use the supplied positive SHAP drivers to explain the model risk.

8. Never say a SHAP feature caused churn.

9. Never invent complaints, product issues, pricing problems,
   competitors, dissatisfaction, budget problems or customer conversations.

10. Missing data is not evidence of a problem.

11. Do not override the Decision Engine.

12. Return exactly the requested six fields.

13. key_drivers must contain exactly three items.

14. next_steps must contain exactly three items.

15. Return ONLY the final JSON object.

16. Do NOT repeat or reproduce the customer input context.

17. Do NOT return the input fields such as customer, risk, financial,
    health, usage, support, renewal or decision.

18. Do NOT explain your instructions.

19. Do NOT provide chain-of-thought.
"""


# ============================================================
# Prompt construction
# ============================================================

def _build_user_prompt(customer_context: Dict[str, Any]) -> str:

    context_json = json.dumps(
        customer_context,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    risk = customer_context.get("risk", {})
    financial = customer_context.get("financial", {})
    decision = customer_context.get("decision", {})
    model_explanation = customer_context.get("model_explanation", {})

    churn_probability = risk.get("churn_probability")
    risk_tier = risk.get("risk_tier")
    recommended_action = decision.get("recommended_action")
    revenue_at_risk = financial.get("revenue_at_risk")
    expected_save_value = financial.get("expected_save_value")

    positive_drivers = model_explanation.get("positive_drivers", [])

    # Explicitly identify the three strongest SHAP drivers in the prompt.
    top_three_drivers = positive_drivers[:3]

    drivers_text = "\n".join(
        f"- {driver.get('display_name') or driver.get('feature')}: "
        f"value={driver.get('value')}, "
        f"SHAP={driver.get('shap_value')}"
        for driver in top_three_drivers
    )

    return f"""
CUSTOMER DATA
=============

{context_json}


YOUR TASK
=========

Write a concise retention insight for this customer.

The following values are authoritative and MUST NOT be changed:

Churn probability:
{_format_percentage(churn_probability)}

Risk tier:
{risk_tier}

Revenue at Risk:
{_format_inr(revenue_at_risk)}

Expected Save Value:
{_format_inr(expected_save_value)}

Recommended action:
{recommended_action}


STRONGEST POSITIVE SHAP DRIVERS
================================

{drivers_text}

Use these three drivers for key_drivers, in the supplied order.
Do not invent additional drivers.


OUTPUT REQUIREMENTS
===================

Return ONLY the requested six-field JSON object.

Do NOT repeat the CUSTOMER DATA.

The output must contain:

1. risk_summary
   - Briefly describe the supplied churn probability and financial exposure.
   - Use INR (₹) for monetary values.

2. key_drivers
   - Exactly 3 items.
   - Use the three strongest positive SHAP drivers supplied above.
   - Do not append repetitive phrases such as "strongest positive driver"
     to every item.
   - A concise format such as "Health Trend: -0.46" is preferred.

3. recommended_action
   - MUST be exactly:
     {recommended_action}

4. reasoning
   - Briefly explain why the Decision Engine action is appropriate.
   - Use the supplied health, usage, support, renewal and financial signals
     where relevant.
   - Do not claim causality.

5. priority
   - MUST be exactly:
     {risk_tier}

6. next_steps
   - Exactly 3 practical Customer Success / Account Management steps.
   - Ground them in the supplied Decision Engine signals.
   - If a signal says there is no significant concern, do not invent a
     problem in that area.
   - Do not invent unsupported customer problems.

IMPORTANT:

Do not return customer profile fields.

Do not return risk, financial, health, usage, support, renewal,
decision or model_explanation objects.

Do not use "$".

Do not use USD.

Do not change the churn probability.

Do not change the risk tier.

Do not change the recommended action.

Do not reproduce the input context.

Return only the final answer.
"""


# ============================================================
# JSON parsing and validation
# ============================================================

def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """Parse and validate Qwen's structured JSON response."""

    if not response_text or not response_text.strip():
        raise RuntimeError("Qwen returned an empty response.")

    text = response_text.strip()

    # --------------------------------------------------------
    # Remove markdown JSON fences if present
    # --------------------------------------------------------

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:1000]
        raise RuntimeError(
            "Qwen returned invalid JSON.\n\n"
            f"Raw response preview:\n{preview}"
        ) from exc

    # --------------------------------------------------------
    # Validate object
    # --------------------------------------------------------

    if not isinstance(result, dict):
        raise RuntimeError("Qwen JSON response must be an object.")

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_fields = {
        "risk_summary",
        "key_drivers",
        "recommended_action",
        "reasoning",
        "priority",
        "next_steps",
    }

    missing_fields = required_fields - set(result.keys())

    if missing_fields:
        raise RuntimeError(
            "Qwen response is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    # Reject unexpected fields as an additional deterministic guard.
    unexpected_fields = set(result.keys()) - required_fields

    if unexpected_fields:
        raise RuntimeError(
            "Qwen response contains unexpected fields: "
            + ", ".join(sorted(unexpected_fields))
        )

    # --------------------------------------------------------
    # Validate string fields
    # --------------------------------------------------------

    string_fields = {
        "risk_summary",
        "recommended_action",
        "reasoning",
        "priority",
    }

    for field in string_fields:
        if not isinstance(result[field], str):
            raise RuntimeError(f"{field} must be a string.")

    # --------------------------------------------------------
    # Validate list fields
    # --------------------------------------------------------

    if not isinstance(result["key_drivers"], list):
        raise RuntimeError("key_drivers must be a list.")

    if not isinstance(result["next_steps"], list):
        raise RuntimeError("next_steps must be a list.")

    # --------------------------------------------------------
    # Validate exact list lengths
    # --------------------------------------------------------

    if len(result["key_drivers"]) != 3:
        raise RuntimeError("key_drivers must contain exactly 3 items.")

    if len(result["next_steps"]) != 3:
        raise RuntimeError("next_steps must contain exactly 3 items.")

    # --------------------------------------------------------
    # Validate list item types
    # --------------------------------------------------------

    for item in result["key_drivers"]:
        if not isinstance(item, str):
            raise RuntimeError(
                "Every key_drivers item must be a string."
            )

    for item in result["next_steps"]:
        if not isinstance(item, str):
            raise RuntimeError(
                "Every next_steps item must be a string."
            )

    # --------------------------------------------------------
    # Validate priority vocabulary
    # --------------------------------------------------------

    allowed_priorities = {
        "Low",
        "Moderate",
        "High",
        "Critical",
    }

    if result["priority"] not in allowed_priorities:
        raise RuntimeError(
            "priority must be one of: Low, Moderate, High, Critical."
        )

    return result


# ============================================================
# Generate insight
# ============================================================

def generate_customer_insight(
    customer_context: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Generate a grounded retention insight using local Ollama/Qwen.

    Qwen is ONLY the natural-language generation layer.

    Prediction:
        XGBoost

    Explanation:
        SHAP

    Business prioritisation:
        Decision Engine
    """

    user_prompt = _build_user_prompt(customer_context)

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt + "\n\n/no_think",
            },
        ],
        # Use an actual JSON schema rather than merely asking for generic JSON.
        format=LLM_OUTPUT_SCHEMA,
        options={
            "temperature": 0.0,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
        think=False,
    )

    # --------------------------------------------------------
    # Extract response
    # --------------------------------------------------------

    message = response.get("message", {})
    response_text = message.get("content", "")

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not response_text.strip():
        thinking = message.get("thinking", "")
        raise RuntimeError(
            "Qwen returned an empty content response. "
            f"Thinking output length: {len(thinking)}."
        )

    # --------------------------------------------------------
    # Parse + validate
    # --------------------------------------------------------

    result = _parse_json_response(response_text)

    # --------------------------------------------------------
    # Deterministic business-authority checks
    # --------------------------------------------------------

    expected_risk_tier = (
        customer_context
        .get("risk", {})
        .get("risk_tier")
    )

    expected_action = (
        customer_context
        .get("decision", {})
        .get("recommended_action")
    )

    # Priority MUST come from Decision Engine.
    if (
        expected_risk_tier is not None
        and result["priority"] != expected_risk_tier
    ):
        raise RuntimeError(
            "LLM priority does not match Decision Engine risk_tier. "
            f"Expected '{expected_risk_tier}', "
            f"received '{result['priority']}'."
        )

    # Recommended action MUST come from Decision Engine.
    if (
        expected_action is not None
        and result["recommended_action"] != expected_action
    ):
        raise RuntimeError(
            "LLM recommended_action does not match Decision Engine. "
            f"Expected '{expected_action}', "
            f"received '{result['recommended_action']}'."
        )

    # --------------------------------------------------------
    # Currency safety check
    # --------------------------------------------------------

    full_text = json.dumps(
        result,
        ensure_ascii=False,
    )

    if "$" in full_text:
        raise RuntimeError(
            "LLM output contains '$'. "
            "All monetary values must be expressed in INR (₹)."
        )

    if "USD" in full_text.upper():
        raise RuntimeError(
            "LLM output contains USD. "
            "All monetary values must be expressed in INR (₹)."
        )

    return result