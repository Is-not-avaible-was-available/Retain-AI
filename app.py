# ============================================================
# RETAIN-AI
# AI-Powered Customer Retention Decision Platform
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import html


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = DATA_DIR / "features.parquet"

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retain-AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    """
    <style>


    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #F5F7FA;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #0B1220;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #0B1220;
    }

    section[data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #CBD5E1 !important;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1 {
        color: #0F172A !important;
        font-weight: 750 !important;
        letter-spacing: -0.8px;
    }

    h2 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    h3 {
        color: #1E293B !important;
        font-weight: 650 !important;
    }

    p {
        color: #475569;
    }


    /* ========================================================
       CUSTOM CARDS
       ======================================================== */

    .retain-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .retain-card-title {
        color: #0F172A;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .retain-card-subtitle {
        color: #64748B;
        font-size: 0.80rem;
        line-height: 1.5;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        min-height: 118px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .kpi-label {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.055em;
        margin-bottom: 9px;
    }

    .kpi-value {
        color: #0F172A;
        font-size: 1.65rem;
        font-weight: 750;
        line-height: 1.15;
    }

    .kpi-subtitle {
        color: #94A3B8;
        font-size: 0.73rem;
        margin-top: 7px;
    }


    /* ========================================================
       BRAND
       ======================================================== */

    .brand-name {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 0.75rem;
        line-height: 1.45;
        margin-top: 5px;
    }

    .sidebar-label {
        color: #64748B;
        font-size: 0.67rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 24px;
        margin-bottom: 8px;
    }

    .sidebar-stat {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 11px 13px;
        margin-bottom: 8px;
    }

    .sidebar-stat-label {
        color: #64748B;
        font-size: 0.64rem;
        font-weight: 650;
    }

    .sidebar-stat-value {
        color: #F8FAFC;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 3px;
    }

    .sidebar-info {
        color: #94A3B8;
        font-size: 0.72rem;
        line-height: 1.55;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px 26px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .hero-title {
        color: #0F172A;
        font-size: 1.75rem;
        font-weight: 750;
        letter-spacing: -0.7px;
    }

    .hero-subtitle {
        color: #64748B;
        font-size: 0.88rem;
        margin-top: 5px;
    }

    .hero-date {
        color: #94A3B8;
        font-size: 0.73rem;
        margin-top: 12px;
    }


    /* ========================================================
       INSIGHT BOX
       ======================================================== */

    .insight-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 17px 19px;
    }

    .insight-label {
        color: #64748B;
        font-size: 0.69rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .insight-text {
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.65;
        margin-top: 7px;
    }


    /* ========================================================
       RISK BADGES
       ======================================================== */

    .risk-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 700;
    }

    .risk-critical {
        background: #FEE2E2;
        color: #991B1B;
    }

    .risk-high {
        background: #FFEDD5;
        color: #9A3412;
    }

    .risk-moderate {
        background: #FEF3C7;
        color: #92400E;
    }

    .risk-low {
        background: #DCFCE7;
        color: #166534;
    }


    /* ========================================================
       SIGNAL CARDS
       ======================================================== */

    .signal-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        min-height: 245px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .signal-title {
        color: #0F172A;
        font-size: 0.98rem;
        font-weight: 700;
    }

    .signal-subtitle {
        color: #64748B;
        font-size: 0.76rem;
        margin-top: 3px;
        margin-bottom: 15px;
    }

    .signal-row {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 9px;
        padding: 10px 12px;
        margin-bottom: 7px;
    }

    .signal-label {
        color: #64748B;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .signal-value {
        color: #0F172A;
        font-size: 0.90rem;
        font-weight: 700;
        margin-top: 2px;
    }


            /* =========================================================
           CUSTOMER 360 — SHAP
           ========================================================= */
        
        .shap-section {
            margin-top: 10px;
        }
        
        .shap-column {
            background: #ffffff;
            border: 1px solid #e4e9f0;
            border-radius: 16px;
            padding: 22px;
            min-height: 100%;
        }
        
        .shap-column-title {
            display: flex;
            align-items: center;
            gap: 9px;
            font-size: 18px;
            font-weight: 700;
            color: #14213d;
            margin-bottom: 20px;
        }
        
        .shap-icon-risk {
            color: #e54868;
            font-size: 19px;
        }
        
        .shap-icon-safe {
            color: #3b82f6;
            font-size: 19px;
        }
        
        .shap-driver {
            padding: 15px 0 17px 0;
            border-bottom: 1px solid #edf0f4;
        }
        
        .shap-driver:last-child {
            border-bottom: none;
        }
        
        .shap-driver-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        
        .shap-feature {
            font-size: 14px;
            font-weight: 650;
            color: #253858;
        }
        
        .shap-value {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 12px;
            font-weight: 600;
            color: #344054;
            background: #f6f8fa;
            border: 1px solid #e5e9ef;
            padding: 4px 8px;
            border-radius: 6px;
            white-space: nowrap;
        }
        
        .shap-driver-meta {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            margin-bottom: 8px;
            font-size: 12px;
            color: #8a94a6;
        }
        
        .shap-driver-meta span:first-child {
            color: #4b5565;
            font-weight: 600;
        }
        
        .shap-track {
            width: 100%;
            height: 6px;
            background: #edf1f5;
            border-radius: 99px;
            overflow: hidden;
        }
        
        .shap-bar-positive,
        .shap-bar-negative {
            height: 100%;
            border-radius: 99px;
        }
        
        .shap-explanation-note {
            margin-top: 14px;
            font-size: 12px;
            color: #8a94a6;
        }

    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.70rem;
        line-height: 1.6;
        padding-top: 35px;
    }

    /* ========================================================
       CUSTOMER 360 — AI RETENTION INSIGHT
       ======================================================== */

    .ai-insight-card {
        background: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }

    .ai-insight-kicker {
        color: #64748B;
        font-size: 0.68rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .ai-insight-title {
        color: #0F172A;
        font-size: 1.15rem;
        font-weight: 750;
        margin-top: 4px;
    }

    .ai-insight-summary {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-top: 12px;
    }

    .ai-insight-label {
        color: #64748B;
        font-size: 0.68rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 7px;
    }

    </style>
    """
)


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_currency(value):

    if pd.isna(value):
        return "₹0"

    value = float(value)

    if abs(value) >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"

    if abs(value) >= 1_00_000:
        return f"₹{value / 1_00_000:.2f} L"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.1f}K"

    return f"₹{value:,.0f}"


def format_probability(value):

    if pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.1f}%"


def format_number(value):

    if pd.isna(value):
        return "N/A"

    return f"{float(value):,.0f}"


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

    if high_burden == 1 or high_escalation == 1:
        return "Technical / CX Escalation"

    if low_csat == 1 or slow_resolution == 1:
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

    if pd.notna(days) and days <= 90:
        actions.append(
            "Renewal Engagement"
        )

    if not actions:
        actions.append(
            "Proactive Customer Monitoring"
        )

    return " • ".join(actions)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(
    show_spinner="Loading feature store..."
)
def load_features():

    if not FEATURES_PATH.exists():

        raise FileNotFoundError(
            f"""
Feature store not found.

Expected location:
{FEATURES_PATH}
"""
        )

    df = pd.read_parquet(
        FEATURES_PATH
    )

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    return df


@st.cache_data(
    show_spinner="Preparing portfolio..."
)
def prepare_portfolio(features):

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

    # IMPORTANT:
    # Do not expose actual churn labels
    # in the production/dashboard layer.

    portfolio = portfolio.drop(
        columns=[
            "churn_90d"
        ],
        errors="ignore"
    )

    return portfolio, latest_date


# ============================================================
# MODEL INFERENCE
# ============================================================

@st.cache_data(
    show_spinner="Running calibrated XGBoost..."
)
def generate_predictions(portfolio):

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


# ============================================================
# BUSINESS LAYER
# ============================================================

@st.cache_data(
    show_spinner="Calculating business risk..."
)
def prepare_business_layer(
    predictions
):

    result = predictions.copy()

    # --------------------------------------------------------
    # REVENUE AT RISK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RISK TIER
    # --------------------------------------------------------

    result[
        "risk_tier"
    ] = (
        result[
            "churn_probability"
        ]
        .apply(
            risk_tier
        )
    )

    # --------------------------------------------------------
    # EXPOSURE TIER
    # --------------------------------------------------------

    thresholds = {

        "moderate":
            result[
                "revenue_at_risk"
            ].quantile(0.50),

        "high":
            result[
                "revenue_at_risk"
            ].quantile(0.75),

        "critical":
            result[
                "revenue_at_risk"
            ].quantile(0.90),
    }

    result[
        "exposure_tier"
    ] = (
        result[
            "revenue_at_risk"
        ]
        .apply(
            lambda x:
            exposure_tier(
                x,
                thresholds
            )
        )
    )

    # --------------------------------------------------------
    # RENEWAL
    # --------------------------------------------------------

    result[
        "renewal_urgency"
    ] = (
        result[
            "days_to_renewal"
        ]
        .apply(
            renewal_urgency
        )
    )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    result[
        "health_signal"
    ] = result.apply(
        health_signal,
        axis=1
    )

    result[
        "usage_signal"
    ] = result.apply(
        usage_signal,
        axis=1
    )

    result[
        "support_signal"
    ] = result.apply(
        support_signal,
        axis=1
    )

    result[
        "recommended_action"
    ] = result.apply(
        recommended_action,
        axis=1
    )

    # --------------------------------------------------------
    # BUSINESS ASSUMPTIONS
    # --------------------------------------------------------

    success_rate = 0.30
    intervention_cost = 25_000

    result[
        "expected_save_value"
    ] = (
        result[
            "revenue_at_risk"
        ]
        * success_rate
    )

    result[
        "intervention_cost"
    ] = intervention_cost

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

    result[
        "expected_roi"
    ] = (
        result[
            "expected_save_value"
        ]
        /
        intervention_cost
    )

    return result


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

try:

    features = load_features()

    portfolio, latest_date = (
        prepare_portfolio(
            features
        )
    )

    predictions = (
        generate_predictions(
            portfolio
        )
    )

    predictions = (
        prepare_business_layer(
            predictions
        )
    )

except Exception as e:

    st.error(
        "Retain-AI could not initialize."
    )

    st.exception(e)

    st.stop()


# ============================================================
# GLOBAL METRICS
# ============================================================

total_customers = len(
    predictions
)

total_acv = (
    predictions[
        "annual_contract_value"
    ]
    .fillna(0)
    .sum()
)

total_revenue_at_risk = (
    predictions[
        "revenue_at_risk"
    ].sum()
)

high_critical_count = len(
    predictions[
        predictions[
            "risk_tier"
        ].isin(
            [
                "High",
                "Critical"
            ]
        )
    ]
)

portfolio_exposure = (
    total_revenue_at_risk
    /
    total_acv
    if total_acv > 0
    else 0
)

average_probability = (
    predictions[
        "churn_probability"
    ].mean()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="padding:10px 2px 18px 2px;">

            <div class="brand-name">
                🎯 Retain-AI
            </div>

            <div class="brand-subtitle">
                Customer Retention<br>
                Intelligence Platform
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="sidebar-label">
            Workspace
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Customer Risk Explorer",
            "Customer 360",
            "Intervention Planner",
        ],
        label_visibility="collapsed",
    )

    st.html(
        """
        <div class="sidebar-label">
            Portfolio
        </div>
        """
    )

    st.html(
        f"""
        <div class="sidebar-stat">

            <div class="sidebar-stat-label">
                CUSTOMERS MONITORED
            </div>

            <div class="sidebar-stat-value">
                {total_customers:,}
            </div>

        </div>

        <div class="sidebar-stat">

            <div class="sidebar-stat-label">
                REVENUE AT RISK
            </div>

            <div class="sidebar-stat-value">
                {format_currency(
                    total_revenue_at_risk
                )}
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="sidebar-label">
            Model
        </div>

        <div class="sidebar-info">

            <b>Model</b><br>
            Calibrated XGBoost<br><br>

            <b>Prediction</b><br>
            90-day churn probability<br><br>

            <b>Business Priority</b><br>
            Revenue at Risk

        </div>
        """
    )

    st.html(
        """
        <div class="sidebar-label">
            Snapshot
        </div>
        """
    )

    st.caption(
        latest_date.strftime(
            "%d %b %Y"
        )
    )

    st.html(
        """
        <div class="sidebar-info"
             style="margin-top:18px;">

            Predictions are generated using
            the production-calibrated XGBoost model.

            <br><br>

            Revenue-at-Risk and intervention
            economics are business decision metrics.

        </div>
        """
    )


# ============================================================
# APPLICATION HERO
# ============================================================

st.html(
    f"""
    <div class="hero">

        <div class="hero-title">
            Retain-AI
        </div>

        <div class="hero-subtitle">
            AI-powered customer retention decision platform
        </div>

        <div class="hero-date">
            Model inference snapshot:
            <b>{latest_date.strftime("%d %B %Y")}</b>
        </div>

    </div>
    """
)


# ============================================================
# CUSTOMER 360 UI HELPERS
# ============================================================


def format_shap_value(value):
    """Format SHAP contribution."""

    value = float(value)

    return f"{value:+.3f}"


def shap_bar_width(shap_value, max_abs_shap):
    """
    Convert SHAP magnitude into a percentage width.
    """

    if max_abs_shap <= 0:
        return 5

    width = (
        abs(float(shap_value))
        / max_abs_shap
        * 100
    )

    return max(6, min(width, 100))


def render_shap_driver(
    driver,
    max_abs_shap,
    positive=True
):
    """
    Render one SHAP driver as a polished HTML card.
    """

    feature = html.escape(
        str(driver["feature"])
    )

    value = html.escape(
        str(driver["value"])
    )

    shap_value = float(
        driver["shap_value"]
    )

    shap_display = format_shap_value(
        shap_value
    )

    width = shap_bar_width(
        shap_value,
        max_abs_shap
    )

    if positive:

        accent = "#E54868"
        bar_class = "shap-bar-positive"
        symbol = "↑"

    else:

        accent = "#3B82F6"
        bar_class = "shap-bar-negative"
        symbol = "↓"

    return f"""
    <div class="shap-driver">

        <div class="shap-driver-header">

            <div class="shap-feature">
                {symbol}&nbsp;&nbsp;{feature}
            </div>

            <div class="shap-value">
                {shap_display}
            </div>

        </div>

        <div class="shap-driver-meta">
            <span>{value}</span>
            <span>SHAP contribution</span>
        </div>

        <div class="shap-track">

            <div
                class="{bar_class}"
                style="
                    width:{width}%;
                    background:{accent};
                "
            ></div>

        </div>

    </div>
    """


# ============================================================
# PAGE 1
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header(
        "Executive Overview"
    )

    st.write(
        "Understand where customer churn risk is concentrated, "
        "how much revenue is exposed, and which accounts deserve "
        "attention."
    )

    st.write("")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Customers Monitored
                </div>

                <div class="kpi-value">
                    {total_customers:,}
                </div>

                <div class="kpi-subtitle">
                    Latest portfolio snapshot
                </div>

            </div>
            """
        )

    with k2:

        st.html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Revenue at Risk
                </div>

                <div class="kpi-value">
                    {format_currency(
                        total_revenue_at_risk
                    )}
                </div>

                <div class="kpi-subtitle">
                    P(churn) × annual contract value
                </div>

            </div>
            """
        )

    with k3:

        st.html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    High / Critical
                </div>

                <div class="kpi-value">
                    {high_critical_count:,}
                </div>

                <div class="kpi-subtitle">
                    {high_critical_count / total_customers * 100:.1f}% of portfolio
                </div>

            </div>
            """
        )

    with k4:

        st.html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Portfolio Exposure
                </div>

                <div class="kpi-value">
                    {portfolio_exposure * 100:.2f}%
                </div>

                <div class="kpi-subtitle">
                    Revenue at Risk / total ACV
                </div>

            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Customer Risk Distribution"
        )

        st.caption(
            "Customers by predicted 90-day churn probability."
        )

        risk_order = [
            "Critical",
            "High",
            "Moderate",
            "Low",
        ]

        risk_counts = (
            predictions[
                "risk_tier"
            ]
            .value_counts()
            .reindex(
                risk_order,
                fill_value=0
            )
        )

        st.bar_chart(
            risk_counts,
            height=310
        )

    with right:

        st.subheader(
            "Revenue at Risk by Segment"
        )

        st.caption(
            "Economic exposure across customer segments."
        )

        segment_risk = (
            predictions
            .groupby(
                "segment"
            )[
                "revenue_at_risk"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            segment_risk,
            height=310
        )

    # --------------------------------------------------------
    # EXECUTIVE INSIGHT
    # --------------------------------------------------------

    enterprise = predictions[
        predictions[
            "segment"
        ] == "Enterprise"
    ]

    enterprise_risk = (
        enterprise[
            "revenue_at_risk"
        ].sum()
    )

    enterprise_risk_share = (
        enterprise_risk
        /
        total_revenue_at_risk
        if total_revenue_at_risk > 0
        else 0
    )

    enterprise_customer_share = (
        len(enterprise)
        /
        total_customers
    )

    st.html(
        f"""
        <div class="insight-box">

            <div class="insight-label">
                Executive Insight
            </div>

            <div class="insight-text">

                <b>Enterprise represents
                {enterprise_customer_share * 100:.1f}%
                of customers but accounts for
                {enterprise_risk_share * 100:.1f}%
                of Revenue at Risk.</b>

                <br><br>

                Retention capacity should therefore
                be allocated using economic exposure,
                not customer count alone.

            </div>

        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # SEGMENT SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Portfolio Risk by Segment"
    )

    segment_summary = (
        predictions
        .groupby(
            "segment"
        )
        .agg(
            customers=(
                "customer_id",
                "count"
            ),
            total_acv=(
                "annual_contract_value",
                "sum"
            ),
            avg_churn_probability=(
                "churn_probability",
                "mean"
            ),
            revenue_at_risk=(
                "revenue_at_risk",
                "sum"
            ),
        )
        .reset_index()
    )

    segment_summary[
        "risk_share"
    ] = (
        segment_summary[
            "revenue_at_risk"
        ]
        /
        total_revenue_at_risk
    )

    display_segment = (
        segment_summary.copy()
    )

    display_segment[
        "total_acv"
    ] = (
        display_segment[
            "total_acv"
        ]
        .map(
            format_currency
        )
    )

    display_segment[
        "avg_churn_probability"
    ] = (
        display_segment[
            "avg_churn_probability"
        ]
        .map(
            format_probability
        )
    )

    display_segment[
        "revenue_at_risk"
    ] = (
        display_segment[
            "revenue_at_risk"
        ]
        .map(
            format_currency
        )
    )

    display_segment[
        "risk_share"
    ] = (
        display_segment[
            "risk_share"
        ]
        .map(
            lambda x:
            f"{x * 100:.1f}%"
        )
    )

    display_segment = (
        display_segment.rename(
            columns={
                "segment":
                    "Segment",
                "customers":
                    "Customers",
                "total_acv":
                    "Annual Contract Value",
                "avg_churn_probability":
                    "Avg. Churn Risk",
                "revenue_at_risk":
                    "Revenue at Risk",
                "risk_share":
                    "Risk Share",
            }
        )
    )

    st.dataframe(
        display_segment,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # TOP ACCOUNTS
    # --------------------------------------------------------

    st.subheader(
        "Highest Revenue-at-Risk Accounts"
    )

    top_accounts = (
        predictions
        .sort_values(
            "revenue_at_risk",
            ascending=False
        )
        .head(10)
        .copy()
    )

    top_display = (
        top_accounts[
            [
                "customer_id",
                "segment",
                "churn_probability",
                "annual_contract_value",
                "revenue_at_risk",
                "risk_tier",
                "behaviour",
                "days_to_renewal",
            ]
        ]
        .copy()
    )

    top_display[
        "churn_probability"
    ] = (
        top_display[
            "churn_probability"
        ]
        .map(
            format_probability
        )
    )

    top_display[
        "annual_contract_value"
    ] = (
        top_display[
            "annual_contract_value"
        ]
        .map(
            format_currency
        )
    )

    top_display[
        "revenue_at_risk"
    ] = (
        top_display[
            "revenue_at_risk"
        ]
        .map(
            format_currency
        )
    )

    top_display = (
        top_display.rename(
            columns={
                "customer_id":
                    "Customer",
                "segment":
                    "Segment",
                "churn_probability":
                    "Churn Risk",
                "annual_contract_value":
                    "ACV",
                "revenue_at_risk":
                    "Revenue at Risk",
                "risk_tier":
                    "Risk",
                "behaviour":
                    "Behaviour",
                "days_to_renewal":
                    "Renewal Days",
            }
        )
    )

    st.dataframe(
        top_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE 2
# CUSTOMER RISK EXPLORER
# ============================================================

elif page == "Customer Risk Explorer":

    st.header(
        "Customer Risk Explorer"
    )

    st.write(
        "Filter and prioritise customers using predictive risk, "
        "economic exposure, behaviour and renewal context."
    )

    st.write("")

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "Search Customer",
        placeholder="Enter customer ID, e.g. C04303",
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:

        segments = sorted(
            predictions[
                "segment"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_segments = st.multiselect(
            "Segment",
            segments,
            default=segments,
        )

    with f2:

        risk_options = [
            "Critical",
            "High",
            "Moderate",
            "Low",
        ]

        selected_risk = st.multiselect(
            "Risk Tier",
            risk_options,
            default=risk_options,
        )

    with f3:

        exposure_options = [
            "Critical",
            "High",
            "Moderate",
            "Low",
        ]

        selected_exposure = st.multiselect(
            "Exposure Tier",
            exposure_options,
            default=exposure_options,
        )

    with f4:

        behaviours = sorted(
            predictions[
                "behaviour"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_behaviour = st.multiselect(
            "Behaviour",
            behaviours,
            default=behaviours,
        )

    with f5:

        renewal_options = [
            "Critical",
            "High",
            "Moderate",
            "Low",
            "Very Low",
            "Unknown",
        ]

        selected_renewal = st.multiselect(
            "Renewal Urgency",
            renewal_options,
            default=renewal_options,
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    filtered = predictions[
        predictions[
            "segment"
        ].isin(
            selected_segments
        )
        &
        predictions[
            "risk_tier"
        ].isin(
            selected_risk
        )
        &
        predictions[
            "exposure_tier"
        ].isin(
            selected_exposure
        )
        &
        predictions[
            "behaviour"
        ].isin(
            selected_behaviour
        )
        &
        predictions[
            "renewal_urgency"
        ].isin(
            selected_renewal
        )
    ].copy()

    if search:

        search_value = (
            search
            .strip()
            .upper()
        )

        filtered = filtered[
            filtered[
                "customer_id"
            ]
            .astype(str)
            .str.upper()
            .str.contains(
                search_value,
                regex=False
            )
        ]

    # --------------------------------------------------------
    # FILTERED KPIs
    # --------------------------------------------------------

    filtered_risk = (
        filtered[
            "revenue_at_risk"
        ].sum()
    )

    filtered_average_risk = (
        filtered[
            "churn_probability"
        ].mean()
        if len(filtered)
        else 0
    )

    filtered_high = len(
        filtered[
            filtered[
                "risk_tier"
            ].isin(
                [
                    "High",
                    "Critical"
                ]
            )
        ]
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        st.metric(
            "Matching Customers",
            f"{len(filtered):,}"
        )

    with a2:

        st.metric(
            "Revenue at Risk",
            format_currency(
                filtered_risk
            )
        )

    with a3:

        st.metric(
            "Average Churn Risk",
            format_probability(
                filtered_average_risk
            )
        )

    with a4:

        st.metric(
            "High / Critical",
            f"{filtered_high:,}"
        )

    st.write("")

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "Priority Accounts"
    )

    display_columns = [
        "customer_id",
        "segment",
        "churn_probability",
        "risk_tier",
        "annual_contract_value",
        "revenue_at_risk",
        "days_to_renewal",
        "behaviour",
        "recommended_action",
    ]

    table = (
        filtered
        .sort_values(
            "revenue_at_risk",
            ascending=False
        )[
            display_columns
        ]
        .copy()
    )

    table[
        "churn_probability"
    ] = (
        table[
            "churn_probability"
        ]
        .map(
            format_probability
        )
    )

    table[
        "annual_contract_value"
    ] = (
        table[
            "annual_contract_value"
        ]
        .map(
            format_currency
        )
    )

    table[
        "revenue_at_risk"
    ] = (
        table[
            "revenue_at_risk"
        ]
        .map(
            format_currency
        )
    )

    table = (
        table.rename(
            columns={
                "customer_id":
                    "Customer",
                "segment":
                    "Segment",
                "churn_probability":
                    "Churn Risk",
                "risk_tier":
                    "Risk",
                "annual_contract_value":
                    "ACV",
                "revenue_at_risk":
                    "Revenue at Risk",
                "days_to_renewal":
                    "Renewal Days",
                "behaviour":
                    "Behaviour",
                "recommended_action":
                    "Recommended Action",
            }
        )
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=560,
    )


# ============================================================
# PAGE 3
# CUSTOMER 360
# ============================================================

elif page == "Customer 360":

    st.header(
        "Customer 360"
    )

    st.write(
        "A complete view of customer risk, commercial exposure, "
        "health, engagement and recommended intervention."
    )

    st.write("")

    # --------------------------------------------------------
    # CUSTOMER SELECTOR
    # --------------------------------------------------------

    customer_ids = (
        predictions[
            "customer_id"
        ]
        .astype(str)
        .sort_values()
        .tolist()
    )

    default_customer = (
        "C04303"
        if "C04303" in customer_ids
        else customer_ids[0]
    )

    selected_customer = st.selectbox(
        "Customer",
        customer_ids,
        index=customer_ids.index(
            default_customer
        ),
    )

    customer = predictions[
        predictions[
            "customer_id"
        ].astype(str)
        ==
        selected_customer
    ].iloc[0]

    st.write("")

    # --------------------------------------------------------
    # CUSTOMER HEADER
    # --------------------------------------------------------

    badge_class = {
        "Critical":
            "risk-critical",
        "High":
            "risk-high",
        "Moderate":
            "risk-moderate",
        "Low":
            "risk-low",
    }.get(
        customer["risk_tier"],
        "risk-low"
    )

    st.html(
        f"""
        <div class="retain-card">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:flex-start;
            ">

                <div>

                    <div style="
                        color:#0F172A;
                        font-size:1.35rem;
                        font-weight:750;
                    ">
                        {selected_customer}
                    </div>

                    <div style="
                        color:#64748B;
                        font-size:0.80rem;
                        margin-top:5px;
                    ">
                        {customer["segment"]}
                        &nbsp; • &nbsp;
                        {customer["industry"]}
                        &nbsp; • &nbsp;
                        {customer["region"]}
                    </div>

                </div>

                <div class="risk-badge {badge_class}">
                    {customer["risk_tier"]}
                </div>

            </div>

        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # CUSTOMER KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Churn Probability",
            format_probability(
                customer[
                    "churn_probability"
                ]
            ),
            help="Predicted probability of churn within the model horizon."
        )

    with c2:

        st.metric(
            "Revenue at Risk",
            format_currency(
                customer[
                    "revenue_at_risk"
                ]
            ),
            help="Churn probability multiplied by annual contract value."
        )

    with c3:

        st.metric(
            "Annual Contract Value",
            format_currency(
                customer[
                    "annual_contract_value"
                ]
            )
        )

    with c4:

        days = customer[
            "days_to_renewal"
        ]

        st.metric(
            "Days to Renewal",
            (
                f"{int(days):,}"
                if pd.notna(days)
                else "N/A"
            )
        )

    st.write("")

    # --------------------------------------------------------
    # THREE SIGNAL CARDS
    # --------------------------------------------------------

    s1, s2, s3 = st.columns(3)

    with s1:

        health_score = (
            f'{customer["health_score"]:.1f}'
            if pd.notna(
                customer["health_score"]
            )
            else "N/A"
        )

        health_trend = (
            f'{customer["health_trend"]:.3f}'
            if pd.notna(
                customer["health_trend"]
            )
            else "N/A"
        )

        st.html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Customer Health
                </div>

                <div class="signal-subtitle">
                    Current health and trajectory
                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Health Score
                    </div>

                    <div class="signal-value">
                        {health_score}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Health Trend
                    </div>

                    <div class="signal-value">
                        {health_trend}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Behaviour
                    </div>

                    <div class="signal-value">
                        {customer["behaviour"]}
                    </div>

                </div>

                <div style="
                    color:#64748B;
                    font-size:0.72rem;
                    margin-top:12px;
                ">
                    <b>Signal:</b>
                    {customer["health_signal"]}
                </div>

            </div>
            """
        )

    with s2:

        active_users = (
            customer[
                "active_users"
            ]
        )

        sessions = (
            customer[
                "sessions"
            ]
        )

        products = (
            customer[
                "product_count"
            ]
        )

        st.html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Product Engagement
                </div>

                <div class="signal-subtitle">
                    Usage and product adoption
                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Active Users
                    </div>

                    <div class="signal-value">
                        {active_users:,.0f}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Sessions
                    </div>

                    <div class="signal-value">
                        {sessions:,.0f}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Products Adopted
                    </div>

                    <div class="signal-value">
                        {products:.0f}
                    </div>

                </div>

                <div style="
                    color:#64748B;
                    font-size:0.72rem;
                    margin-top:12px;
                ">
                    <b>Signal:</b>
                    {customer["usage_signal"]}
                </div>

            </div>
            """
        )

    with s3:

        csat = customer[
            "avg_csat_12w"
        ]

        csat_value = (
            f"{csat:.2f}"
            if pd.notna(csat)
            else "No data"
        )

        st.html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Customer Experience
                </div>

                <div class="signal-subtitle">
                    Support interaction profile
                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Tickets — 12W
                    </div>

                    <div class="signal-value">
                        {customer["tickets_12w"]:.0f}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Escalations — 12W
                    </div>

                    <div class="signal-value">
                        {customer["escalations_12w"]:.0f}
                    </div>

                </div>

                <div class="signal-row">

                    <div class="signal-label">
                        Average CSAT — 12W
                    </div>

                    <div class="signal-value">
                        {csat_value}
                    </div>

                </div>

                <div style="
                    color:#64748B;
                    font-size:0.72rem;
                    margin-top:12px;
                ">
                    <b>Signal:</b>
                    {customer["support_signal"]}
                </div>

            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # RETENTION RECOMMENDATION
    # --------------------------------------------------------

    st.subheader(
        "Retention Recommendation"
    )

    st.success(
        customer[
            "recommended_action"
        ]
    )

    st.caption(
        "The recommendation combines customer-health, "
        "usage, support and renewal signals. "
        "Business priority is determined separately using Revenue at Risk."
    )

    st.write("")

    # --------------------------------------------------------
    # BUSINESS CONTEXT
    # --------------------------------------------------------

    b1, b2, b3 = st.columns(3)

    with b1:

        st.metric(
            "Expected Save Value",
            format_currency(
                customer[
                    "expected_save_value"
                ]
            )
        )

    with b2:

        st.metric(
            "Intervention Cost",
            format_currency(
                customer[
                    "intervention_cost"
                ]
            )
        )

    with b3:

        st.metric(
            "Expected Benefit / Cost",
            f'{customer["expected_roi"]:.2f}×'
        )

    st.caption(
        "Economics assume a 30% intervention success rate "
        "and ₹25,000 intervention cost."
    )

    st.write("")

    # --------------------------------------------------------
    # AI RETENTION INSIGHT
    # --------------------------------------------------------

    st.subheader(
        "AI Retention Insight"
    )

    st.caption(
        "Qwen3:4B translates the calibrated XGBoost prediction, "
        "SHAP explanation and Decision Engine signals into an "
        "action-oriented retention brief. It does not make the churn prediction."
    )

    if "ai_insights" not in st.session_state:
        st.session_state["ai_insights"] = {}

    generate_ai_insight = st.button(
        "Generate AI Retention Insight",
        type="primary",
        use_container_width=False,
        key="generate_ai_retention_insight",
    )

    if generate_ai_insight:

        try:

            from decision_engine import build_decision_engine
            from explainability import explain_customer
            from llm_insights import (
                prepare_customer_context,
                generate_customer_insight,
            )

            with st.spinner(
                "Preparing model explanation and generating AI insight..."
            ):

                # ------------------------------------------------
                # SHAP explanation
                # ------------------------------------------------

                customer_row = (
                    portfolio[
                        portfolio["customer_id"].astype(str)
                        == str(selected_customer)
                    ]
                    .iloc[[0]]
                    .copy()
                )

                explanation_for_ai = explain_customer(
                    customer_row,
                    top_n=5
                )

                # ------------------------------------------------
                # Decision Engine — use the full portfolio so
                # priority/selection remain portfolio-relative.
                # ------------------------------------------------

                decisions_for_ai = build_decision_engine(
                    predictions=predictions,
                    intervention_success_rate=0.30,
                    intervention_cost=25_000,
                    intervention_capacity=0.10,
                )

                decision_customer = (
                    decisions_for_ai[
                        decisions_for_ai["customer_id"].astype(str)
                        == str(selected_customer)
                    ]
                    .iloc[0]
                )

                # ------------------------------------------------
                # Structured context → local LLM
                # ------------------------------------------------

                llm_context = prepare_customer_context(
                    customer,
                    decision_customer,
                    explanation_for_ai
                )

                insight = generate_customer_insight(
                    llm_context
                )

                st.session_state["ai_insights"][
                    str(selected_customer)
                ] = insight

        except Exception as e:

            st.error(
                "Unable to generate the AI retention insight. "
                "Make sure Ollama is running and Qwen3:4B is available."
            )
            st.exception(e)

    # --------------------------------------------------------
    # DISPLAY CACHED INSIGHT
    # --------------------------------------------------------

    ai_insight = st.session_state["ai_insights"].get(
        str(selected_customer)
    )

    if ai_insight:

        st.html(
            f"""
            <div class="ai-insight-card">

                <div class="ai-insight-kicker">
                    Local AI • Qwen3:4B • Model-grounded
                </div>

                <div class="ai-insight-title">
                    Retention Brief for {html.escape(str(selected_customer))}
                </div>

                <div class="ai-insight-summary">
                    {html.escape(str(ai_insight["risk_summary"]))}
                </div>

            </div>
            """
        )

        st.write("")

        ai_priority, ai_action = st.columns(2, gap="large")

        with ai_priority:

            priority_class = {
                "Critical": "risk-critical",
                "High": "risk-high",
                "Moderate": "risk-moderate",
                "Low": "risk-low",
            }.get(
                str(ai_insight["priority"]),
                "risk-low",
            )

            st.html(
                f"""
                <div class="retain-card">
                    <div class="ai-insight-label">AI Priority</div>
                    <span class="risk-badge {priority_class}">
                        {html.escape(str(ai_insight["priority"]))}
                    </span>
                </div>
                """
            )

        with ai_action:

            st.html(
                f"""
                <div class="retain-card">
                    <div class="ai-insight-label">Recommended Action</div>
                    <div class="retain-card-title">
                        {html.escape(str(ai_insight["recommended_action"]))}
                    </div>
                </div>
                """
            )

        st.write("")

        ai1, ai2 = st.columns(2, gap="large")

        driver_items = "".join(
            f"<li>{html.escape(str(driver))}</li>"
            for driver in ai_insight["key_drivers"]
        )

        step_items = "".join(
            f"<li>{html.escape(str(step))}</li>"
            for step in ai_insight["next_steps"]
        )

        with ai1:

            st.html(
                f"""
                <div class="retain-card">
                    <div class="ai-insight-label">Key Risk Drivers</div>
                    <ul style="
                        margin:8px 0 0 18px;
                        padding:0;
                        color:#334155;
                        line-height:1.7;
                        font-size:0.86rem;
                    ">
                        {driver_items}
                    </ul>
                </div>
                """
            )

        with ai2:

            st.html(
                f"""
                <div class="retain-card">
                    <div class="ai-insight-label">Next Steps</div>
                    <ol style="
                        margin:8px 0 0 18px;
                        padding:0;
                        color:#334155;
                        line-height:1.7;
                        font-size:0.86rem;
                    ">
                        {step_items}
                    </ol>
                </div>
                """
            )

        st.write("")

        st.html(
            f"""
            <div class="retain-card">
                <div class="ai-insight-label">Why This Matters</div>
                <div class="retain-card-subtitle" style="
                    font-size:0.86rem;
                    color:#334155;
                    line-height:1.65;
                ">
                    {html.escape(str(ai_insight["reasoning"]))}
                </div>
            </div>
            """
        )

        st.caption(
            "AI-generated communication layer. Churn probability, SHAP explanations "
            "and retention recommendations remain governed by the production model "
            "and Decision Engine."
        )

    else:

        st.info(
            "Select a customer and click **Generate AI Retention Insight** "
            "to create a grounded retention brief."
        )

    st.write("")

    # --------------------------------------------------------
    # SHAP AREA
    # --------------------------------------------------------

    # ============================================================
    # SHAP EXPLANATION
    # ============================================================

    st.header("Why is this customer at risk?")

    st.caption(
        "SHAP explains which customer attributes are "
        "pushing the model prediction higher or lower."
    )

    st.write("")
    
    try:
    
        from explainability import explain_customer
    
        # --------------------------------------------------------
        # Customer row
        # --------------------------------------------------------
    
        customer_row = (
            portfolio[
                portfolio["customer_id"].astype(str)
                == str(selected_customer)
            ]
            .iloc[[0]]
            .copy()
        )
    
        # --------------------------------------------------------
        # SHAP
        # --------------------------------------------------------
    
        explanation = explain_customer(
            customer_row,
            top_n=5
        )
    
        positive_drivers = (
            explanation["positive_drivers"]
        )
    
        negative_drivers = (
            explanation["negative_drivers"]
        )
    
        # --------------------------------------------------------
        # Determine common scale
        # --------------------------------------------------------
    
        all_drivers = (
            positive_drivers
            + negative_drivers
        )
    
        max_abs_shap = max(
            [
                abs(
                    float(x["shap_value"])
                )
                for x in all_drivers
            ],
            default=1
        )
    
        # --------------------------------------------------------
        # Render cards
        # --------------------------------------------------------
    
        left, right = st.columns(
            2,
            gap="large"
        )
    
        # ========================================================
        # INCREASING RISK
        # ========================================================
    
        with left:
        
            positive_html = ""
    
            for driver in positive_drivers:
            
                positive_html += render_shap_driver(
                    driver,
                    max_abs_shap,
                    positive=True
                )
    
            if not positive_html:
            
                positive_html = """
                <div style="
                    color:#8a94a6;
                    font-size:14px;
                    padding:20px 0;
                ">
                    No significant risk-increasing
                    drivers identified.
                </div>
                """
    
            st.html(
                f"""
                <div class="shap-column">
    
                    <div class="shap-column-title">
                        <span class="shap-icon-risk">▲</span>
                        Increasing churn risk
                    </div>
    
                    {positive_html}
    
                    <div class="shap-explanation-note">
                        Positive SHAP values increase the
                        model's predicted churn risk.
                    </div>
    
                </div>
                """
            )
    
        # ========================================================
        # REDUCING RISK
        # ========================================================
    
        with right:
        
            negative_html = ""
    
            for driver in negative_drivers:
            
                negative_html += render_shap_driver(
                    driver,
                    max_abs_shap,
                    positive=False
                )
    
            if not negative_html:
            
                negative_html = """
                <div style="
                    color:#8a94a6;
                    font-size:14px;
                    padding:20px 0;
                ">
                    No significant risk-reducing
                    drivers identified.
                </div>
                """
    
            st.html(
                f"""
                <div class="shap-column">
    
                    <div class="shap-column-title">
                        <span class="shap-icon-safe">▼</span>
                        Reducing churn risk
                    </div>
    
                    {negative_html}
    
                    <div class="shap-explanation-note">
                        Negative SHAP values reduce the
                        model's predicted churn risk.
                    </div>
    
                </div>
                """
            )
    
    
    except Exception as e:
    
        st.error(
            "Unable to generate customer explanation."
        )
    
        st.exception(e)

# ============================================================
# PAGE 4
# INTERVENTION PLANNER
# ============================================================

elif page == "Intervention Planner":

    st.header(
        "Intervention Planner"
    )

    st.write(
        "Allocate limited Customer Success capacity to the "
        "customers where intervention is expected to create "
        "the greatest economic value."
    )

    st.write("")

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    st.subheader(
        "Intervention Capacity"
    )

    st.caption(
        "Select the percentage of the portfolio your team can actively engage."
    )

    capacity_options = [
        "1%",
        "2%",
        "5%",
        "10%",
        "15%",
        "20%",
    ]

    capacity_map = {
        "1%": 0.01,
        "2%": 0.02,
        "5%": 0.05,
        "10%": 0.10,
        "15%": 0.15,
        "20%": 0.20,
    }

    selected_capacity_label = st.select_slider(
        "Capacity",
        options=capacity_options,
        value="10%",
        label_visibility="collapsed",
    )

    selected_capacity = capacity_map[
        selected_capacity_label
    ]

    st.write("")

    # --------------------------------------------------------
    # DECISION ENGINE
    # --------------------------------------------------------

    from decision_engine import (
        build_decision_engine
    )

    decisions = build_decision_engine(
        predictions=predictions,
        intervention_success_rate=0.30,
        intervention_cost=25_000,
        intervention_capacity=selected_capacity,
    )

    selected = decisions[
        decisions[
            "selected_for_intervention"
        ]
    ].copy()

    total_portfolio_risk = (
        decisions[
            "revenue_at_risk"
        ].sum()
    )

    selected_risk = (
        selected[
            "revenue_at_risk"
        ].sum()
    )

    risk_coverage = (
        selected_risk
        /
        total_portfolio_risk
        if total_portfolio_risk > 0
        else 0
    )

    expected_save = (
        selected[
            "expected_save_value"
        ].sum()
    )

    intervention_cost = (
        selected[
            "intervention_cost"
        ].sum()
    )

    net_expected_value = (
        selected[
            "net_expected_value"
        ].sum()
    )

    expected_roi = (
        expected_save
        /
        intervention_cost
        if intervention_cost > 0
        else 0
    )

    # --------------------------------------------------------
    # PLANNER KPIs
    # --------------------------------------------------------

    p1, p2, p3, p4, p5 = st.columns(5)

    planner_data = [

        (
            "Customers Selected",
            f"{len(selected):,}",
            f"{selected_capacity_label} of portfolio"
        ),

        (
            "Risk Coverage",
            f"{risk_coverage * 100:.1f}%",
            "Revenue at Risk covered"
        ),

        (
            "Expected Save",
            format_currency(
                expected_save
            ),
            "30% success assumption"
        ),

        (
            "Net Expected Value",
            format_currency(
                net_expected_value
            ),
            "Expected save − cost"
        ),

        (
            "Expected Benefit / Cost",
            f"{expected_roi:.2f}×",
            "Expected benefit / cost"
        ),
    ]

    for column, values in zip(
        [
            p1,
            p2,
            p3,
            p4,
            p5,
        ],
        planner_data
    ):

        with column:

            st.html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        {values[0]}
                    </div>

                    <div class="kpi-value">
                        {values[1]}
                    </div>

                    <div class="kpi-subtitle">
                        {values[2]}
                    </div>

                </div>
                """
            )

    st.write("")

    st.info(
        """
        Expected Benefit / Cost assumes a 30% intervention success rate
        and ₹25,000 intervention cost per customer.
        These are explicit business assumptions, not
        model-learned values.
        """
    )

    st.write("")

    # --------------------------------------------------------
    # RECOMMENDED ALLOCATION
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="insight-box">

            <div class="insight-label">
                Recommended Allocation
            </div>

            <div class="insight-text">

                With capacity to intervene with
                <b>{selected_capacity_label}</b>
                of the portfolio, Retain-AI recommends
                engaging <b>{len(selected):,} customers</b>.

                This group covers
                <b>{risk_coverage * 100:.1f}%</b>
                of total Revenue at Risk and represents
                <b>{format_currency(expected_save)}</b>
                of expected save value.

            </div>

        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # PRIORITY TABLE
    # --------------------------------------------------------

    st.subheader(
        "Intervention Priority List"
    )

    priority_columns = [
        "priority_rank",
        "customer_id",
        "segment",
        "churn_probability",
        "annual_contract_value",
        "revenue_at_risk",
        "expected_save_value",
        "net_expected_value",
        "risk_tier",
        "days_to_renewal",
        "behaviour",
        "recommended_action",
    ]

    priority = (
        selected
        .sort_values(
            "priority_rank"
        )[
            priority_columns
        ]
        .head(100)
        .copy()
    )

    priority[
        "churn_probability"
    ] = (
        priority[
            "churn_probability"
        ]
        .map(
            format_probability
        )
    )

    for column in [
        "annual_contract_value",
        "revenue_at_risk",
        "expected_save_value",
        "net_expected_value",
    ]:

        priority[
            column
        ] = (
            priority[
                column
            ]
            .map(
                format_currency
            )
        )

    priority = (
        priority.rename(
            columns={
                "priority_rank":
                    "Priority",
                "customer_id":
                    "Customer",
                "segment":
                    "Segment",
                "churn_probability":
                    "Churn Risk",
                "annual_contract_value":
                    "ACV",
                "revenue_at_risk":
                    "Revenue at Risk",
                "expected_save_value":
                    "Expected Save",
                "net_expected_value":
                    "Net Expected Value",
                "risk_tier":
                    "Risk",
                "days_to_renewal":
                    "Renewal Days",
                "behaviour":
                    "Behaviour",
                "recommended_action":
                    "Recommended Action",
            }
        )
    )

    st.dataframe(
        priority,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.write("")

    # --------------------------------------------------------
    # PRIORITISATION LOGIC
    # --------------------------------------------------------

    st.subheader(
        "How Retain-AI Prioritises Customers"
    )

    st.markdown(
        """
        **1. Predict churn probability**

        The calibrated XGBoost model estimates the probability
        of customer churn.

        **2. Calculate Revenue at Risk**

        `Revenue at Risk = P(churn) × Annual Contract Value`

        **3. Estimate expected save value**

        `Expected Save Value = Revenue at Risk × 30%`

        **4. Account for intervention cost**

        `Net Expected Value = Expected Save Value − ₹25,000`

        **5. Prioritise**

        Customers are ranked by expected business value rather
        than churn probability alone.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="app-footer">

        <b>Retain-AI</b>
        &nbsp;•&nbsp;
        Customer Retention Intelligence

        <br>

        Calibrated XGBoost
        &nbsp;•&nbsp;
        Explainable AI
        &nbsp;•&nbsp;
        Business Decision Engine

    </div>
    """
)   