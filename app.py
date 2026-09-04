import sys
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Retain-AI — Post Business Diagnostic",
    page_icon="🔬",
    layout="wide",
)

st.title("Retain-AI — Post Business Layer Diagnostic")

st.write(
    "Testing the remaining Executive Overview pipeline:"
    " Decision Engine → Historical Inference → Aggregation → Charts"
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
# HELPER
# ============================================================

def run_step(number, title, function):

    st.subheader(f"{number}. {title}")

    start = time.perf_counter()

    try:

        result = function()

        elapsed = time.perf_counter() - start

        st.success(
            f"{title} PASSED — {elapsed:.2f} seconds"
        )

        return result

    except Exception as e:

        st.error(
            f"{title} FAILED"
        )

        st.exception(e)

        st.stop()


# ============================================================
# 1. LOAD FEATURES
# ============================================================

def load_features():

    df = pd.read_parquet(
        FEATURES_PATH
    )

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    return df


features = run_step(
    "1",
    "Load Feature Store",
    load_features,
)

st.write(
    "Feature store shape:",
    features.shape
)

st.write(
    "Memory usage:",
    f"{features.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
)


# ============================================================
# 2. PREPARE LATEST PORTFOLIO
# ============================================================

def prepare_portfolio():

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

    return portfolio, latest_date


portfolio, latest_date = run_step(
    "2",
    "Prepare Latest Portfolio",
    prepare_portfolio,
)

st.write(
    "Latest snapshot:",
    latest_date
)

st.write(
    "Portfolio shape:",
    portfolio.shape
)


# ============================================================
# 3. LATEST XGBOOST INFERENCE
# ============================================================

def latest_inference():

    from inference import (
        predict_churn_probability
    )

    probabilities = (
        predict_churn_probability(
            portfolio
        )
    )

    result = portfolio.copy()

    result[
        "churn_probability"
    ] = probabilities

    return result


predictions = run_step(
    "3",
    "Latest XGBoost Inference",
    latest_inference,
)

st.write(
    "Predictions:",
    len(predictions)
)


# ============================================================
# 4. BUSINESS LAYER
# ============================================================

def business_layer():

    result = predictions.copy()

    result[
        "revenue_at_risk"
    ] = (
        result[
            "churn_probability"
        ]
        *
        result[
            "annual_contract_value"
        ].fillna(0)
    )

    result[
        "risk_tier"
    ] = (
        result[
            "churn_probability"
        ].apply(
            lambda x:
                "Critical"
                if x >= 0.20
                else
                "High"
                if x >= 0.10
                else
                "Moderate"
                if x >= 0.05
                else
                "Low"
        )
    )

    result[
        "expected_save_value"
    ] = (
        result[
            "revenue_at_risk"
        ]
        * 0.30
    )

    result[
        "intervention_cost"
    ] = 25_000

    result[
        "net_expected_value"
    ] = (
        result[
            "expected_save_value"
        ]
        -
        result[
            "intervention_cost"
        ]
    )

    return result


predictions = run_step(
    "4",
    "Business Layer",
    business_layer,
)


# ============================================================
# 5. DECISION ENGINE
# ============================================================

def decision_engine():

    from decision_engine import (
        build_decision_engine
    )

    return build_decision_engine(
        predictions=predictions,
        intervention_success_rate=0.30,
        intervention_cost=25_000,
        intervention_capacity=0.10,
    )


decisions = run_step(
    "5",
    "Decision Engine",
    decision_engine,
)

st.write(
    "Decision dataset shape:",
    decisions.shape
)

if "selected_for_intervention" in decisions.columns:

    st.write(
        "Selected for intervention:",
        int(
            decisions[
                "selected_for_intervention"
            ].sum()
        )
    )


# ============================================================
# 6. PREPARE HISTORICAL DATA
# ============================================================

def prepare_historical_data():

    historical = (
        features
        .drop(
            columns=[
                "churn_90d"
            ],
            errors="ignore"
        )
        .copy()
    )

    return historical


historical = run_step(
    "6",
    "Prepare Historical Inference Dataset",
    prepare_historical_data,
)

st.write(
    "Historical dataset shape:",
    historical.shape
)

st.write(
    "Historical dataset memory:",
    f"{historical.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
)


# ============================================================
# 7. HISTORICAL XGBOOST — CRITICAL TEST
# ============================================================

st.subheader(
    "7. Historical XGBoost Inference — CRITICAL TEST"
)

st.warning(
    "This is the operation most likely to expose the Streamlit "
    "Cloud memory/process problem. It runs the production model "
    "across the full feature store."
)

st.write(
    f"Rows to score: {len(historical):,}"
)

st.write(
    "Expected snapshots:",
    historical["snapshot_date"].nunique()
)

st.write(
    "Expected customers:",
    historical["customer_id"].nunique()
)

st.write(
    "Starting historical model inference NOW..."
)

start_time = time.perf_counter()

try:

    from inference import (
        predict_churn_probability
    )

    historical_probabilities = (
        predict_churn_probability(
            historical
        )
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    st.success(
        "Historical XGBoost inference PASSED."
    )

    st.write(
        "Elapsed time:",
        f"{elapsed:.2f} seconds"
    )

    st.write(
        "Predictions:",
        len(historical_probabilities)
    )

    st.write(
        "Min probability:",
        float(
            np.min(
                historical_probabilities
            )
        )
    )

    st.write(
        "Max probability:",
        float(
            np.max(
                historical_probabilities
            )
        )
    )

    st.write(
        "Mean probability:",
        float(
            np.mean(
                historical_probabilities
            )
        )
    )

except Exception as e:

    st.error(
        "Historical XGBoost inference FAILED."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 8. BUILD HISTORICAL PORTFOLIO
# ============================================================

def build_historical_portfolio():

    result = historical.copy()

    result[
        "churn_probability"
    ] = historical_probabilities

    result[
        "revenue_at_risk"
    ] = (
        result[
            "churn_probability"
        ]
        *
        result[
            "annual_contract_value"
        ].fillna(0)
    )

    return result


historical_predictions = run_step(
    "8",
    "Build Historical Portfolio Predictions",
    build_historical_portfolio,
)

st.write(
    "Historical prediction shape:",
    historical_predictions.shape
)


# ============================================================
# 9. MONTHLY RISK AGGREGATION
# ============================================================

def monthly_risk():

    monthly = (
        historical_predictions
        .set_index(
            "snapshot_date"
        )[
            "churn_probability"
        ]
        .resample("MS")
        .mean()
        .dropna()
        .to_frame(
            "Average predicted churn risk"
        )
    )

    return monthly


monthly_risk_df = run_step(
    "9",
    "Monthly Risk Aggregation",
    monthly_risk,
)

st.write(
    "Monthly observations:",
    len(monthly_risk_df)
)

st.dataframe(
    monthly_risk_df,
    use_container_width=True,
)


# ============================================================
# 10. MONTHLY REVENUE AT RISK
# ============================================================

def monthly_exposure():

    monthly = (
        historical_predictions
        .set_index(
            "snapshot_date"
        )[
            "revenue_at_risk"
        ]
        .resample("MS")
        .sum()
        .dropna()
        .to_frame(
            "Revenue at Risk"
        )
    )

    return monthly


monthly_exposure_df = run_step(
    "10",
    "Monthly Revenue-at-Risk Aggregation",
    monthly_exposure,
)

st.write(
    "Monthly observations:",
    len(monthly_exposure_df)
)

st.dataframe(
    monthly_exposure_df,
    use_container_width=True,
)


# ============================================================
# 11. TREND CALCULATIONS
# ============================================================

def trend_calculations():

    trend_start = (
        historical_predictions[
            "snapshot_date"
        ].min()
    )

    trend_end = (
        historical_predictions[
            "snapshot_date"
        ].max()
    )

    latest_month = (
        monthly_risk_df
        .iloc[-1, 0]
    )

    first_month = (
        monthly_risk_df
        .iloc[0, 0]
    )

    risk_change = (
        latest_month
        -
        first_month
    )

    return (
        trend_start,
        trend_end,
        latest_month,
        first_month,
        risk_change,
    )


(
    trend_start,
    trend_end,
    latest_month,
    first_month,
    risk_change,
) = run_step(
    "11",
    "Trend Calculations",
    trend_calculations,
)

st.write(
    "Trend window:",
    f"{trend_start} → {trend_end}"
)

st.write(
    "Risk change:",
    f"{risk_change * 100:+.2f} percentage points"
)


# ============================================================
# 12. STREAMLIT CHART TEST
# ============================================================

st.subheader(
    "12. Streamlit Chart Rendering"
)

try:

    st.write(
        "Testing average predicted churn risk chart..."
    )

    st.line_chart(
        monthly_risk_df,
        height=280,
    )

    st.success(
        "Average predicted churn risk chart PASSED."
    )

except Exception as e:

    st.error(
        "Average predicted churn risk chart FAILED."
    )

    st.exception(e)


try:

    st.write(
        "Testing Revenue-at-Risk chart..."
    )

    st.line_chart(
        monthly_exposure_df,
        height=280,
    )

    st.success(
        "Revenue-at-Risk chart PASSED."
    )

except Exception as e:

    st.error(
        "Revenue-at-Risk chart FAILED."
    )

    st.exception(e)


# ============================================================
# 13. FINAL
# ============================================================

st.subheader(
    "13. FINAL DIAGNOSTIC RESULT"
)

st.success(
    "POST-BUSINESS-LAYER DIAGNOSTIC COMPLETED."
)

st.write(
    "If this entire page loads, the crash is NOT caused by:"
)

st.write(
    """
    • Decision Engine
    • Historical XGBoost inference
    • Historical aggregation
    • Trend calculations
    • Streamlit line charts
    """
)

st.write(
    "The remaining issue will therefore be in one of the "
    "other Executive Overview / Customer Risk Explorer / "
    "Customer 360 / Intervention Planner rendering blocks."
)