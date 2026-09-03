# ============================================================
# RETAIN-AI — BUSINESS DECISION ENGINE
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# DEFAULT BUSINESS ASSUMPTIONS
# ============================================================

DEFAULT_INTERVENTION_SUCCESS_RATE = 0.30
DEFAULT_INTERVENTION_COST = 25_000
DEFAULT_INTERVENTION_CAPACITY = 0.10


# ============================================================
# RISK TIERS
# ============================================================

def assign_risk_tier(
    probability: float
) -> str:
    """
    Convert calibrated churn probability into
    an operational risk tier.
    """

    if probability < 0.05:
        return "Low"

    elif probability < 0.10:
        return "Moderate"

    elif probability < 0.20:
        return "High"

    return "Critical"


# ============================================================
# FINANCIAL EXPOSURE
# ============================================================

def assign_exposure_tier(
    revenue_at_risk: float,
    p50: float,
    p75: float,
    p90: float
) -> str:
    """
    Assign financial exposure based on portfolio
    Revenue-at-Risk percentiles.
    """

    if revenue_at_risk < p50:
        return "Low"

    elif revenue_at_risk < p75:
        return "Moderate"

    elif revenue_at_risk < p90:
        return "High"

    return "Critical"


# ============================================================
# RENEWAL URGENCY
# ============================================================

def assign_renewal_urgency(
    days_to_renewal: float
) -> str:
    """
    Convert days-to-renewal into an operational
    urgency category.
    """

    if days_to_renewal <= 30:
        return "Critical"

    elif days_to_renewal <= 90:
        return "High"

    elif days_to_renewal <= 180:
        return "Moderate"

    elif days_to_renewal <= 365:
        return "Low"

    return "Very Low"


# ============================================================
# HEALTH SIGNAL
# ============================================================

def assign_health_signal(
    health_score: float,
    health_trend: float,
    behaviour: str
) -> str:
    """
    Identify the strongest account-health condition.
    """

    if (
        health_score < 40
        or health_trend < -0.20
        or behaviour == "Rapidly Declining"
    ):
        return "Critical deterioration"

    elif (
        health_score < 60
        or health_trend < -0.05
        or behaviour == "Declining"
    ):
        return "Deteriorating"

    elif (
        health_score >= 75
        and health_trend > 0.05
    ):
        return "Healthy / Improving"

    return "Stable"


# ============================================================
# USAGE SIGNAL
# ============================================================

def assign_usage_signal(
    row: pd.Series
) -> str:
    """
    Identify the strongest product engagement signal.
    """

    if row["severe_usage_decline_12w_flag"] == 1:
        return "Severe persistent usage decline"

    elif row["usage_decline_12w_flag"] == 1:
        return "Persistent usage decline"

    elif row["usage_decline_4w_flag"] == 1:
        return "Recent usage decline"

    return "No significant usage decline"


# ============================================================
# SUPPORT SIGNAL
# ============================================================

def assign_support_signal(
    row: pd.Series
) -> str:
    """
    Identify customer-support / experience condition.

    Missing CSAT is not interpreted as poor CSAT.
    """

    if (
        row["high_escalation_12w"] == 1
        or row["high_support_burden_12w"] == 1
    ):
        return "High support burden"

    elif (
        row["low_csat_12w"] == 1
        or row["slow_resolution_12w"] == 1
    ):
        return "Customer experience concern"

    return "No significant support concern"


# ============================================================
# INTERVENTION RECOMMENDATIONS
# ============================================================

def generate_interventions(
    row: pd.Series
) -> list[str]:
    """
    Generate evidence-based intervention themes.

    The decision engine recommends intervention categories
    based on observable business signals.

    It does not claim that an intervention will definitely
    prevent churn.
    """

    interventions = []

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    if (
        row["health_score"] < 40
        or row["health_trend"] < -0.20
        or row["behaviour"] == "Rapidly Declining"
    ):
        interventions.append(
            "Account Health / Customer Success Intervention"
        )

    elif (
        row["health_score"] < 60
        or row["health_trend"] < -0.05
        or row["behaviour"] == "Declining"
    ):
        interventions.append(
            "Customer Success Monitoring"
        )

    # --------------------------------------------------------
    # PRODUCT ENGAGEMENT
    # --------------------------------------------------------

    if row["severe_usage_decline_12w_flag"] == 1:
        interventions.append(
            "Urgent Product Adoption Intervention"
        )

    elif row["usage_decline_12w_flag"] == 1:
        interventions.append(
            "Product Adoption / Engagement Intervention"
        )

    elif row["usage_decline_4w_flag"] == 1:
        interventions.append(
            "Early Engagement Warning"
        )

    # --------------------------------------------------------
    # SUPPORT / EXPERIENCE
    # --------------------------------------------------------

    if (
        row["high_support_burden_12w"] == 1
        or row["high_escalation_12w"] == 1
    ):
        interventions.append(
            "Technical / Customer Experience Escalation"
        )

    elif (
        row["low_csat_12w"] == 1
        or row["slow_resolution_12w"] == 1
    ):
        interventions.append(
            "Customer Experience Review"
        )

    # --------------------------------------------------------
    # RENEWAL
    # --------------------------------------------------------

    if row["days_to_renewal"] <= 30:
        interventions.append(
            "Immediate Renewal Engagement"
        )

    elif row["days_to_renewal"] <= 90:
        interventions.append(
            "Renewal Engagement"
        )

    elif (
        row["days_to_renewal"] <= 180
        and row["churn_probability"] >= 0.10
    ):
        interventions.append(
            "Proactive Renewal Planning"
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not interventions:
        interventions.append("Monitor")

    return interventions


# ============================================================
# BUILD DECISION DATASET
# ============================================================

def build_decision_engine(
    predictions: pd.DataFrame,
    intervention_success_rate: float = (
        DEFAULT_INTERVENTION_SUCCESS_RATE
    ),
    intervention_cost: float = (
        DEFAULT_INTERVENTION_COST
    ),
    intervention_capacity: float = (
        DEFAULT_INTERVENTION_CAPACITY
    )
) -> pd.DataFrame:
    """
    Convert churn predictions into business decisions.

    Parameters
    ----------
    predictions:
        DataFrame containing customer-level predictions and
        required business features.

    intervention_success_rate:
        Assumed probability that an intervention successfully
        saves an otherwise-at-risk account.

    intervention_cost:
        Assumed cost per intervention.

    intervention_capacity:
        Fraction of customers the business can intervene on.

    Returns
    -------
    pd.DataFrame
        Customer-level business decision dataset.
    """

    if not 0 < intervention_success_rate <= 1:
        raise ValueError(
            "intervention_success_rate must be between 0 and 1."
        )

    if intervention_cost < 0:
        raise ValueError(
            "intervention_cost cannot be negative."
        )

    if not 0 < intervention_capacity <= 1:
        raise ValueError(
            "intervention_capacity must be between 0 and 1."
        )

    required_columns = [
        "customer_id",
        "churn_probability",
        "annual_contract_value",
        "days_to_renewal",
        "health_score",
        "health_trend",
        "behaviour",
        "usage_decline_4w_flag",
        "usage_decline_12w_flag",
        "severe_usage_decline_12w_flag",
        "high_support_burden_12w",
        "high_escalation_12w",
        "low_csat_12w",
        "slow_resolution_12w"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df = predictions.copy()

    # ========================================================
    # 1. RISK
    # ========================================================

    df["risk_tier"] = (
        df["churn_probability"]
        .apply(assign_risk_tier)
    )

    # ========================================================
    # 2. REVENUE AT RISK
    # ========================================================

    df["revenue_at_risk"] = (
        df["churn_probability"]
        * df["annual_contract_value"]
    )

    # ========================================================
    # 3. EXPECTED SAVE
    # ========================================================

    df["expected_save_value"] = (
        df["revenue_at_risk"]
        * intervention_success_rate
    )

    # ========================================================
    # 4. INTERVENTION COST
    # ========================================================

    df["intervention_cost"] = intervention_cost

    # ========================================================
    # 5. NET EXPECTED VALUE
    # ========================================================

    df["net_expected_value"] = (
        df["expected_save_value"]
        - df["intervention_cost"]
    )

    # ========================================================
    # 6. BENEFIT / COST
    # ========================================================

    df["expected_benefit_cost_ratio"] = np.where(
        df["intervention_cost"] > 0,
        (
            df["expected_save_value"]
            / df["intervention_cost"]
        ),
        np.inf
    )

    # ========================================================
    # 7. EXPECTED ROI
    # ========================================================

    df["expected_roi"] = np.where(
        df["intervention_cost"] > 0,
        (
            df["net_expected_value"]
            / df["intervention_cost"]
        ),
        np.inf
    )

    # ========================================================
    # 8. FINANCIAL EXPOSURE
    # ========================================================

    p50 = df[
        "revenue_at_risk"
    ].quantile(0.50)

    p75 = df[
        "revenue_at_risk"
    ].quantile(0.75)

    p90 = df[
        "revenue_at_risk"
    ].quantile(0.90)

    df["exposure_tier"] = (
        df["revenue_at_risk"]
        .apply(
            lambda value:
            assign_exposure_tier(
                value,
                p50,
                p75,
                p90
            )
        )
    )

    # ========================================================
    # 9. RENEWAL URGENCY
    # ========================================================

    df["renewal_urgency"] = (
        df["days_to_renewal"]
        .apply(assign_renewal_urgency)
    )

    # ========================================================
    # 10. HEALTH SIGNAL
    # ========================================================

    df["health_signal"] = df.apply(
        lambda row:
        assign_health_signal(
            row["health_score"],
            row["health_trend"],
            row["behaviour"]
        ),
        axis=1
    )

    # ========================================================
    # 11. USAGE SIGNAL
    # ========================================================

    df["usage_signal"] = df.apply(
        assign_usage_signal,
        axis=1
    )

    # ========================================================
    # 12. SUPPORT SIGNAL
    # ========================================================

    df["support_signal"] = df.apply(
        assign_support_signal,
        axis=1
    )

    # ========================================================
    # 13. INTERVENTIONS
    # ========================================================

    df["recommended_interventions"] = df.apply(
        generate_interventions,
        axis=1
    )

    df["recommended_action"] = (
        df["recommended_interventions"]
        .apply(
            lambda actions:
            " | ".join(actions)
        )
    )

    # ========================================================
    # 14. ECONOMIC VIABILITY
    # ========================================================

    df["economically_viable"] = (
        df["net_expected_value"] > 0
    )

    # ========================================================
    # 15. PRIORITY
    # ========================================================

    df["priority_rank"] = (
        df["net_expected_value"]
        .rank(
            ascending=False,
            method="first"
        )
        .astype(int)
    )

    # ========================================================
    # 16. INTERVENTION CAPACITY
    # ========================================================

    capacity_count = max(
        1,
        int(
            np.floor(
                len(df)
                * intervention_capacity
            )
        )
    )

    eligible = (
        df[
            df["economically_viable"]
        ]
        .sort_values(
            "net_expected_value",
            ascending=False
        )
    )

    selected_ids = set(
        eligible
        .head(capacity_count)[
            "customer_id"
        ]
    )

    df["selected_for_intervention"] = (
        df["customer_id"]
        .isin(selected_ids)
    )

    # ========================================================
    # FINAL ORDER
    # ========================================================

    df = (
        df.sort_values(
            "priority_rank"
        )
        .reset_index(drop=True)
    )

    return df