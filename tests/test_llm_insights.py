from pathlib import Path
import json
import sys

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# Load project data
# ============================================================

features = pd.read_parquet(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

features["snapshot_date"] = pd.to_datetime(
    features["snapshot_date"]
)


# ============================================================
# Latest portfolio snapshot
# ============================================================

latest_date = features["snapshot_date"].max()

portfolio = features[
    features["snapshot_date"] == latest_date
].copy()


portfolio = portfolio.drop(
    columns=["churn_90d"],
    errors="ignore"
)


# ============================================================
# Select customer
# ============================================================

customer = portfolio[
    portfolio["customer_id"].astype(str) == "C04303"
].iloc[[0]].copy()


if customer.empty:
    raise ValueError(
        "C04303 was not found."
    )


print()
print("=" * 70)
print("CUSTOMER")
print("=" * 70)

print(
    customer[
        [
            "customer_id",
            "segment",
            "health_score",
            "health_trend",
            "behaviour",
            "annual_contract_value",
        ]
    ].to_string(index=False)
)


# ============================================================
# Production inference
# ============================================================

from inference import predict_churn_probability


model_input = customer.drop(
    columns=[
        "customer_id",
        "contract_id",
        "snapshot_date",
    ],
    errors="ignore",
)


probability = predict_churn_probability(
    model_input
)[0]


customer["churn_probability"] = probability


print()
print("=" * 70)
print("CHURN PROBABILITY")
print("=" * 70)

print(
    f"{probability:.6f}"
)

print(
    f"{probability * 100:.2f}%"
)


# ============================================================
# Business Decision Engine
# ============================================================

from decision_engine import build_decision_engine


decision_customer = build_decision_engine(
    customer.copy(),
    intervention_success_rate=0.30,
    intervention_cost=25_000,
    intervention_capacity=0.10,
)


if decision_customer.empty:
    raise ValueError(
        "Decision engine returned an empty result."
    )


# ------------------------------------------------------------
# Preserve one customer record
# ------------------------------------------------------------

customer = decision_customer.copy()


print()
print("=" * 70)
print("DECISION ENGINE")
print("=" * 70)


decision_columns = [
    "customer_id",
    "churn_probability",
    "risk_tier",
    "revenue_at_risk",
    "expected_save_value",
    "intervention_cost",
    "net_expected_value",
    "expected_benefit_cost_ratio",
    "exposure_tier",
    "renewal_urgency",
    "health_signal",
    "usage_signal",
    "support_signal",
    "recommended_interventions",
    "recommended_action",
    "economically_viable",
    "selected_for_intervention",
    "priority_rank",
]


available_decision_columns = [
    column
    for column in decision_columns
    if column in customer.columns
]


print(
    customer[
        available_decision_columns
    ].to_string(index=False)
)


# ============================================================
# SHAP explanation
# ============================================================

from explainability import explain_customer


explanation = explain_customer(
    customer,
    top_n=5,
)


print()
print("=" * 70)
print("SHAP POSITIVE DRIVERS")
print("=" * 70)

for driver in explanation.get(
    "positive_drivers",
    []
):

    print(
        f"{driver.get('display_name')}: "
        f"{driver.get('value')} "
        f"(SHAP={float(driver.get('shap_value')):.4f})"
    )


print()
print("=" * 70)
print("SHAP PROTECTIVE DRIVERS")
print("=" * 70)

for driver in explanation.get(
    "negative_drivers",
    []
):

    print(
        f"{driver.get('display_name')}: "
        f"{driver.get('value')} "
        f"(SHAP={float(driver.get('shap_value')):.4f})"
    )


# ============================================================
# Prepare LLM context
# ============================================================

from llm_insights import (
    prepare_customer_context,
    generate_customer_insight,
)


context = prepare_customer_context(
    customer_row=customer,
    explanation=explanation,
)


print()
print("=" * 70)
print("LLM INPUT CONTEXT")
print("=" * 70)

print(
    json.dumps(
        context,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
)


# ============================================================
# Generate LLM insight
# ============================================================

print()
print("=" * 70)
print("GENERATING QWEN RETENTION INSIGHT")
print("=" * 70)

print(
    "Sending structured model + SHAP + business context to Qwen3:4B..."
)


insight = generate_customer_insight(
    customer_context=context
)


print()
print("=" * 70)
print("LLM INSIGHT")
print("=" * 70)

print(
    json.dumps(
        insight,
        indent=2,
        ensure_ascii=False,
    )
)


# ============================================================
# Basic validation
# ============================================================

required_keys = {
    "risk_summary",
    "key_drivers",
    "recommended_action",
    "reasoning",
    "priority",
    "next_steps",
}


missing_keys = (
    required_keys
    - set(insight.keys())
)


if missing_keys:

    raise AssertionError(
        f"Missing output keys: {missing_keys}"
    )


# ------------------------------------------------------------
# Type validation
# ------------------------------------------------------------

assert isinstance(
    insight["risk_summary"],
    str
)

assert isinstance(
    insight["key_drivers"],
    list
)

assert isinstance(
    insight["recommended_action"],
    str
)

assert isinstance(
    insight["reasoning"],
    str
)

assert isinstance(
    insight["next_steps"],
    list
)

assert insight["priority"] in {
    "Low",
    "Moderate",
    "High",
    "Critical",
}


# ------------------------------------------------------------
# Content validation
# ------------------------------------------------------------

assert len(
    insight["key_drivers"]
) >= 1

assert len(
    insight["next_steps"]
) >= 1


print()
print("=" * 70)
print("✓ LLM INSIGHT TEST PASSED")
print("=" * 70)