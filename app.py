import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Retain-AI Business Layer Diagnostic",
    page_icon="🔬",
    layout="wide",
)

st.title("Retain-AI — Business Layer Diagnostic")

st.write(
    "Testing feature store → portfolio → XGBoost → business layer."
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# BUSINESS FUNCTIONS
# ============================================================

def risk_tier(probability):

    if probability >= 0.20:
        return "Critical"

    if probability >= 0.10:
        return "High"

    if probability >= 0.05:
        return "Moderate"

    return "Low"


def exposure_tier(value, thresholds):

    if value >= thresholds["critical"]:
        return "Critical"

    if value >= thresholds["high"]:
        return "High"

    if value >= thresholds["moderate"]:
        return "Moderate"

    return "Low"


def renewal_urgency(days):

    if pd.isna(days):
        return "Unknown"

    days = float(days)

    if days <= 30:
        return "Critical"

    if days <= 90:
        return "High"

    if days <= 180:
        return "Moderate"

    if days <= 365:
        return "Low"

    return "Very Low"


def health_signal(row):

    health = row.get(
        "health_score",
        np.nan
    )

    trend = row.get(
        "health_trend",
        np.nan
    )

    behaviour = row.get(
        "behaviour",
        ""
    )

    if (
        (pd.notna(health) and health < 40)
        or
        (pd.notna(trend) and trend < -0.20)
        or
        behaviour == "Rapidly Declining"
    ):
        return "Account Health Intervention"

    if (
        (pd.notna(health) and health < 60)
        or
        (pd.notna(trend) and trend < -0.05)
        or
        behaviour == "Declining"
    ):
        return "Customer Success Monitoring"

    return "Healthy"


def usage_signal(row):

    severe = row.get(
        "severe_usage_decline_12w_flag",
        0
    )

    decline_12 = row.get(
        "usage_decline_12w_flag",
        0
    )

    decline_4 = row.get(
        "usage_decline_4w_flag",
        0
    )

    if severe == 1:
        return "Severe Product Engagement Risk"

    if decline_12 == 1:
        return "Product Adoption Intervention"

    if decline_4 == 1:
        return "Early Engagement Warning"

    return "Stable Usage"


def support_signal(row):

    high_burden = row.get(
        "high_support_burden_12w",
        0
    )

    high_escalation = row.get(
        "high_escalation_12w",
        0
    )

    low_csat = row.get(
        "low_csat_12w",
        0
    )

    slow_resolution = row.get(
        "slow_resolution_12w",
        0
    )

    if (
        high_burden == 1
        or high_escalation == 1
    ):
        return "Technical / CX Escalation"

    if (
        low_csat == 1
        or slow_resolution == 1
    ):
        return "Customer Experience Review"

    return "Normal Support Profile"


def recommended_action(row):

    actions = []

    health = row.get(
        "health_score",
        np.nan
    )

    trend = row.get(
        "health_trend",
        np.nan
    )

    behaviour = row.get(
        "behaviour",
        ""
    )

    if (
        (pd.notna(health) and health < 40)
        or
        (pd.notna(trend) and trend < -0.20)
        or
        behaviour == "Rapidly Declining"
    ):
        actions.append(
            "Account Health Review"
        )

    if (
        row.get(
            "severe_usage_decline_12w_flag",
            0
        ) == 1
        or
        row.get(
            "usage_decline_12w_flag",
            0
        ) == 1
    ):
        actions.append(
            "Product Adoption"
        )

    if (
        row.get(
            "high_support_burden_12w",
            0
        ) == 1
        or
        row.get(
            "high_escalation_12w",
            0
        ) == 1
    ):
        actions.append(
            "CX / Technical Escalation"
        )

    days = row.get(
        "days_to_renewal",
        np.nan
    )

    if (
        pd.notna(days)
        and days <= 90
    ):
        actions.append(
            "Renewal Engagement"
        )

    if not actions:
        actions.append(
            "Proactive Customer Monitoring"
        )

    return " • ".join(actions)


# ============================================================
# 1. ENVIRONMENT
# ============================================================

st.subheader("1. Environment")

st.write(
    "Python:",
    sys.version.split()[0]
)

st.write(
    "NumPy:",
    np.__version__
)

st.write(
    "Pandas:",
    pd.__version__
)


# ============================================================
# 2. LOAD FEATURE STORE
# ============================================================

st.subheader("2. Loading feature store")

try:

    features = pd.read_parquet(
        FEATURES_PATH
    )

    features["snapshot_date"] = pd.to_datetime(
        features["snapshot_date"]
    )

    st.success(
        f"Feature store loaded: {features.shape}"
    )

except Exception as e:

    st.error(
        "Feature-store loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 3. PREPARE PORTFOLIO
# ============================================================

st.subheader("3. Preparing latest portfolio")

try:

    latest_date = (
        features[
            "snapshot_date"
        ].max()
    )

    portfolio = (
        features[
            features[
                "snapshot_date"
            ] == latest_date
        ]
        .copy()
        .reset_index(drop=True)
    )

    portfolio = portfolio.drop(
        columns=[
            "churn_90d"
        ],
        errors="ignore"
    )

    st.success(
        "Portfolio prepared."
    )

    st.write(
        "Latest snapshot:",
        latest_date
    )

    st.write(
        "Portfolio shape:",
        portfolio.shape
    )

    st.write(
        "Unique customers:",
        portfolio[
            "customer_id"
        ].nunique()
    )

except Exception as e:

    st.error(
        "Portfolio preparation failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 4. XGBOOST
# ============================================================

st.subheader("4. Running XGBoost")

try:

    from inference import (
        predict_churn_probability
    )

    with st.spinner(
        "Running calibrated XGBoost..."
    ):

        probabilities = (
            predict_churn_probability(
                portfolio
            )
        )

    predictions = portfolio.copy()

    predictions[
        "churn_probability"
    ] = probabilities

    st.success(
        "XGBoost inference completed."
    )

    st.write(
        "Predictions:",
        len(probabilities)
    )

except Exception as e:

    st.error(
        "XGBoost inference failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 5. BUSINESS LAYER
# ============================================================

st.subheader("5. Business Layer")

business = predictions.copy()


# ------------------------------------------------------------
# 5A. REVENUE AT RISK
# ------------------------------------------------------------

st.write("**5A. Revenue at Risk**")

try:

    business[
        "revenue_at_risk"
    ] = (
        business[
            "churn_probability"
        ]
        *
        business[
            "annual_contract_value"
        ].fillna(0)
    )

    st.success(
        "Revenue at Risk calculation passed."
    )

except Exception as e:

    st.error(
        "Revenue at Risk calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5B. RISK TIER
# ------------------------------------------------------------

st.write("**5B. Risk Tier**")

try:

    business[
        "risk_tier"
    ] = (
        business[
            "churn_probability"
        ].apply(
            risk_tier
        )
    )

    st.success(
        "Risk tier calculation passed."
    )

    st.write(
        business[
            "risk_tier"
        ].value_counts()
    )

except Exception as e:

    st.error(
        "Risk tier calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5C. EXPOSURE TIER
# ------------------------------------------------------------

st.write("**5C. Exposure Tier**")

try:

    thresholds = {

        "moderate":
            business[
                "revenue_at_risk"
            ].quantile(0.50),

        "high":
            business[
                "revenue_at_risk"
            ].quantile(0.75),

        "critical":
            business[
                "revenue_at_risk"
            ].quantile(0.90),
    }

    business[
        "exposure_tier"
    ] = (
        business[
            "revenue_at_risk"
        ].apply(
            lambda x:
                exposure_tier(
                    x,
                    thresholds
                )
        )
    )

    st.success(
        "Exposure tier calculation passed."
    )

    st.write(
        "Exposure thresholds:",
        thresholds
    )

    st.write(
        business[
            "exposure_tier"
        ].value_counts()
    )

except Exception as e:

    st.error(
        "Exposure tier calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5D. RENEWAL URGENCY
# ------------------------------------------------------------

st.write("**5D. Renewal Urgency**")

try:

    business[
        "renewal_urgency"
    ] = (
        business[
            "days_to_renewal"
        ].apply(
            renewal_urgency
        )
    )

    st.success(
        "Renewal urgency calculation passed."
    )

    st.write(
        business[
            "renewal_urgency"
        ].value_counts()
    )

except Exception as e:

    st.error(
        "Renewal urgency calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5E. HEALTH SIGNAL
# ------------------------------------------------------------

st.write("**5E. Health Signal**")

try:

    business[
        "health_signal"
    ] = business.apply(
        health_signal,
        axis=1
    )

    st.success(
        "Health signal calculation passed."
    )

    st.write(
        business[
            "health_signal"
        ].value_counts()
    )

except Exception as e:

    st.error(
        "Health signal calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5F. USAGE SIGNAL
# ------------------------------------------------------------

st.write("**5F. Usage Signal**")

try:

    business[
        "usage_signal"
    ] = business.apply(
        usage_signal,
        axis=1
    )

    st.success(
        "Usage signal calculation passed."
    )

    st.write(
        business[
            "usage_signal"
        ].value_counts()
    )

except Exception as e:

    st.error(
        "Usage signal calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5G. SUPPORT SIGNAL
# ------------------------------------------------------------

st.write("**5G. Support Signal**")

try:

    business[
        "support_signal"
    ] = business.apply(
        support_signal,
        axis=1
    )

    st.success(
        "Support signal calculation passed."
    )

    st.write(
        business[
            "support_signal"
        ].value_counts()
    )

except Exception as e:

    st.error(
        "Support signal calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5H. RECOMMENDED ACTION
# ------------------------------------------------------------

st.write("**5H. Recommended Action**")

try:

    business[
        "recommended_action"
    ] = business.apply(
        recommended_action,
        axis=1
    )

    st.success(
        "Recommended action calculation passed."
    )

    st.write(
        business[
            "recommended_action"
        ].value_counts().head(20)
    )

except Exception as e:

    st.error(
        "Recommended action calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5I. EXPECTED SAVE VALUE
# ------------------------------------------------------------

st.write("**5I. Expected Save Value**")

try:

    success_rate = 0.30

    business[
        "expected_save_value"
    ] = (
        business[
            "revenue_at_risk"
        ]
        *
        success_rate
    )

    st.success(
        "Expected save value calculation passed."
    )

except Exception as e:

    st.error(
        "Expected save value calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5J. INTERVENTION COST
# ------------------------------------------------------------

st.write("**5J. Intervention Cost**")

try:

    intervention_cost = 25_000

    business[
        "intervention_cost"
    ] = intervention_cost

    st.success(
        "Intervention cost calculation passed."
    )

except Exception as e:

    st.error(
        "Intervention cost calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5K. NET EXPECTED VALUE
# ------------------------------------------------------------

st.write("**5K. Net Expected Value**")

try:

    business[
        "net_expected_value"
    ] = (
        business[
            "expected_save_value"
        ]
        -
        business[
            "intervention_cost"
        ]
    )

    st.success(
        "Net expected value calculation passed."
    )

except Exception as e:

    st.error(
        "Net expected value calculation failed."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# 5L. EXPECTED ROI
# ------------------------------------------------------------

st.write("**5L. Expected ROI**")

try:

    business[
        "expected_roi"
    ] = (
        business[
            "expected_save_value"
        ]
        /
        intervention_cost
    )

    st.success(
        "Expected ROI calculation passed."
    )

except Exception as e:

    st.error(
        "Expected ROI calculation failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

st.subheader("6. Business Layer Summary")

st.success(
    "ALL BUSINESS-LAYER OPERATIONS PASSED."
)

st.write(
    "Final business dataset shape:",
    business.shape
)

st.write(
    "Total Revenue at Risk:",
    business[
        "revenue_at_risk"
    ].sum()
)

st.write(
    "Total Expected Save Value:",
    business[
        "expected_save_value"
    ].sum()
)

st.write(
    "Total Intervention Cost:",
    business[
        "intervention_cost"
    ].sum()
)

st.write(
    "Total Net Expected Value:",
    business[
        "net_expected_value"
    ].sum()
)


# ============================================================
# SAMPLE OUTPUT
# ============================================================

st.subheader("7. Sample Business Output")

display_columns = [
    "customer_id",
    "segment",
    "annual_contract_value",
    "churn_probability",
    "revenue_at_risk",
    "risk_tier",
    "exposure_tier",
    "renewal_urgency",
    "health_signal",
    "usage_signal",
    "support_signal",
    "recommended_action",
    "expected_save_value",
    "net_expected_value",
]

available_columns = [
    c for c in display_columns
    if c in business.columns
]

st.dataframe(
    business[
        available_columns
    ].head(20),
    use_container_width=True,
)

st.success(
    "Business-layer diagnostic completed successfully."
)