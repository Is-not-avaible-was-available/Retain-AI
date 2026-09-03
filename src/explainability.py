# ============================================================
# RETAIN-AI
# Explainability Layer
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = (
    MODELS_DIR / "xgb_preprocessor.joblib"
)

XGB_MODEL_PATH = (
    MODELS_DIR / "xgb_regularized.joblib"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_explainability_model():
    """
    Load the underlying regularized XGBoost model.

    SHAP explains the underlying tree model rather than
    the calibrated wrapper.
    """

    return joblib.load(
        XGB_MODEL_PATH
    )


def load_explainability_preprocessor():
    """
    Load the exact production feature preprocessor used
    to transform the model inputs.
    """

    return joblib.load(
        PREPROCESSOR_PATH
    )


# ============================================================
# FEATURE NAME MAPPING
# ============================================================

def clean_feature_name(feature_name):
    """
    Convert sklearn ColumnTransformer feature names into
    human-readable business-facing names.

    Actual feature names look like:

        numeric__health_score
        numeric__health_trend
        categorical__behaviour_Declining
        categorical__acquisition_channel_Partner
    """

    name = str(feature_name)

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    if name.startswith("numeric__"):

        original_name = name.replace(
            "numeric__",
            "",
            1
        )

        explicit = {

            "health_trend":
                "Health Trend",

            "health_score":
                "Health Score",

            "baseline_health":
                "Baseline Health",

            "volatility":
                "Health Volatility",

            "company_size":
                "Company Size",

            "tenure_days":
                "Tenure (Days)",

            "tenure_years":
                "Tenure (Years)",

            "annual_contract_value":
                "Annual Contract Value",

            "total_contract_value":
                "Total Contract Value",

            "active_users":
                "Active Users",

            "sessions":
                "Sessions",

            "active_days":
                "Active Days",

            "usage_minutes":
                "Usage Minutes",

            "features_used":
                "Features Used",

            "product_count":
                "Product Count",

            "tickets_12w":
                "Support Tickets (12W)",

            "escalations_12w":
                "Escalations (12W)",

            "reopens_12w":
                "Reopened Tickets (12W)",

            "critical_tickets_12w":
                "Critical Tickets (12W)",

            "high_priority_tickets_12w":
                "High Priority Tickets (12W)",

            "avg_csat_12w":
                "Average CSAT (12W)",

            "avg_resolution_hours_12w":
                "Average Resolution Time (12W)",

            "days_to_renewal":
                "Days to Renewal",

            "health_score_change_4w":
                "Health Change (4W)",

            "health_score_change_8w":
                "Health Change (8W)",

            "health_score_change_12w":
                "Health Change (12W)",

            "health_score_pct_change_4w":
                "Health Change % (4W)",

            "health_score_pct_change_8w":
                "Health Change % (8W)",

            "health_score_pct_change_12w":
                "Health Change % (12W)",

            "active_users_change_4w":
                "Active Users Change (4W)",

            "active_users_change_8w":
                "Active Users Change (8W)",

            "active_users_change_12w":
                "Active Users Change (12W)",

            "sessions_change_4w":
                "Sessions Change (4W)",

            "sessions_change_8w":
                "Sessions Change (8W)",

            "sessions_change_12w":
                "Sessions Change (12W)",

            "usage_minutes_change_4w":
                "Usage Minutes Change (4W)",

            "usage_minutes_change_8w":
                "Usage Minutes Change (8W)",

            "usage_minutes_change_12w":
                "Usage Minutes Change (12W)",

            "usage_decline_4w_flag":
                "Usage Decline (4W)",

            "usage_decline_12w_flag":
                "Usage Decline (12W)",

            "severe_usage_decline_12w_flag":
                "Severe Usage Decline (12W)",

            "high_support_burden_12w":
                "High Support Burden",

            "high_escalation_12w":
                "High Escalation Rate",

            "low_csat_12w":
                "Low CSAT",

            "slow_resolution_12w":
                "Slow Resolution",

            "auto_renew":
                "Auto Renewal",
        }

        return explicit.get(
            original_name,
            original_name.replace(
                "_",
                " "
            ).title()
        )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    if name.startswith("categorical__"):

        original_name = name.replace(
            "categorical__",
            "",
            1
        )

        categorical_prefixes = {

            "acquisition_channel_":
                "Acquisition Channel",

            "industry_":
                "Industry",

            "region_":
                "Region",

            "segment_":
                "Segment",

            "contract_type_":
                "Contract Type",

            "behaviour_":
                "Behaviour",
        }

        for prefix, display_name in categorical_prefixes.items():

            if original_name.startswith(prefix):

                category = original_name[
                    len(prefix):
                ]

                category = category.replace(
                    "_",
                    " "
                )

                return (
                    f"{display_name}: "
                    f"{category}"
                )

        return original_name.replace(
            "_",
            " "
        ).title()

    # Fallback

    return name.replace(
        "_",
        " "
    ).title()


# ============================================================
# ORIGINAL COLUMN FROM TRANSFORMED FEATURE
# ============================================================

def get_original_column(
    transformed_feature_name
):
    """
    Map a transformed feature back to its original
    feature-store column.

    Examples
    --------
    numeric__health_score
        -> health_score

    numeric__annual_contract_value
        -> annual_contract_value

    categorical__behaviour_Declining
        -> behaviour

    categorical__acquisition_channel_Partner
        -> acquisition_channel
    """

    name = str(
        transformed_feature_name
    )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    if name.startswith(
        "numeric__"
    ):

        return name.replace(
            "numeric__",
            "",
            1
        )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    if name.startswith(
        "categorical__"
    ):

        feature = name.replace(
            "categorical__",
            "",
            1
        )

        categorical_columns = [
            "industry",
            "region",
            "segment",
            "acquisition_channel",
            "behaviour",
            "contract_type",
        ]

        for column in categorical_columns:

            prefix = (
                column
                + "_"
            )

            if feature.startswith(
                prefix
            ):

                return column

    return None


# ============================================================
# FEATURE VALUE FORMATTING
# ============================================================

def format_feature_value(
    feature_name,
    value
):
    """
    Convert original feature values into
    readable business-facing values.
    """

    if value is None:
        return "Not available"

    if pd.isna(value):
        return "Not available"

    name = str(
        feature_name
    )

    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------

    if name == "auto_renew":

        return (
            "Yes"
            if bool(value)
            else "No"
        )

    # --------------------------------------------------------
    # Usage / binary flags
    # --------------------------------------------------------

    if name.endswith(
        "_flag"
    ):

        return (
            "Yes"
            if float(value) == 1
            else "No"
        )

    # --------------------------------------------------------
    # Business signal flags
    # --------------------------------------------------------

    if name in [
        "high_support_burden_12w",
        "high_escalation_12w",
        "low_csat_12w",
        "slow_resolution_12w",
    ]:

        return (
            "Yes"
            if float(value) == 1
            else "No"
        )

    # --------------------------------------------------------
    # Monetary values
    # --------------------------------------------------------

    if name in [
        "annual_contract_value",
        "total_contract_value",
    ]:

        return (
            f"₹{float(value):,.0f}"
        )

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    if name in [
        "health_score",
        "baseline_health",
    ]:

        return (
            f"{float(value):.1f}"
        )

    # --------------------------------------------------------
    # Trend / percentage change
    # --------------------------------------------------------

    if (
        "trend" in name
        or "pct_change" in name
    ):

        return (
            f"{float(value):.2f}"
        )

    # --------------------------------------------------------
    # CSAT
    # --------------------------------------------------------

    if "csat" in name:

        return (
            f"{float(value):.2f}"
        )

    # --------------------------------------------------------
    # Resolution time
    # --------------------------------------------------------

    if (
        "resolution_hours"
        in name
    ):

        return (
            f"{float(value):.1f} hrs"
        )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    if (
        "tickets" in name
        or "sessions" in name
        or "users" in name
        or "days" in name
        or "product_count" in name
        or "features_used" in name
        or "active_days" in name
    ):

        return (
            f"{float(value):,.0f}"
        )

    # --------------------------------------------------------
    # General numeric
    # --------------------------------------------------------

    if isinstance(
        value,
        (int, np.integer)
    ):

        return (
            f"{int(value):,}"
        )

    if isinstance(
        value,
        (float, np.floating)
    ):

        return (
            f"{float(value):.3f}"
        )

    return str(value)


# ============================================================
# SHAP EXPLAINER
# ============================================================

def build_shap_explainer():

    model = (
        load_explainability_model()
    )

    return shap.TreeExplainer(
        model
    )


# ============================================================
# SINGLE CUSTOMER EXPLANATION
# ============================================================

def explain_customer(
    customer_row: pd.DataFrame,
    top_n: int = 5,
):
    """
    Generate a SHAP explanation for one customer.

    Parameters
    ----------
    customer_row:
        One-row DataFrame containing the same features
        used by production inference.

    top_n:
        Number of positive and negative drivers.

    Returns
    -------
    dict
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(
        customer_row,
        pd.DataFrame
    ):

        raise TypeError(
            "customer_row must be a pandas DataFrame."
        )

    if len(customer_row) != 1:

        raise ValueError(
            "customer_row must contain exactly one row."
        )

    # --------------------------------------------------------
    # Remove metadata
    # --------------------------------------------------------

    metadata_columns = [
        "customer_id",
        "contract_id",
        "snapshot_date",
        "churn_90d",
    ]

    model_input = (
        customer_row
        .drop(
            columns=metadata_columns,
            errors="ignore"
        )
        .copy()
    )

    # --------------------------------------------------------
    # Load production preprocessor
    # --------------------------------------------------------

    preprocessor = (
        load_explainability_preprocessor()
    )

    transformed = (
        preprocessor.transform(
            model_input
        )
    )

    # --------------------------------------------------------
    # Build SHAP explainer
    # --------------------------------------------------------

    explainer = (
        build_shap_explainer()
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = (
        explainer.shap_values(
            transformed
        )
    )

    # SHAP compatibility
    if isinstance(
        shap_values,
        list
    ):

        shap_values = (
            shap_values[-1]
        )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 2:

        shap_values = (
            shap_values[0]
        )

    # --------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if len(shap_values) != len(
        feature_names
    ):

        raise ValueError(
            "SHAP feature count does not "
            "match preprocessor feature count. "
            f"SHAP={len(shap_values)}, "
            f"features={len(feature_names)}"
        )

    # --------------------------------------------------------
    # Original customer values
    # --------------------------------------------------------

    original_row = (
        customer_row.iloc[0]
    )

    # --------------------------------------------------------
    # Build SHAP records
    # --------------------------------------------------------

    records = []

    for i, feature_name in enumerate(
        feature_names
    ):

        shap_value = float(
            shap_values[i]
        )

        original_column = (
            get_original_column(
                feature_name
            )
        )

        # Retrieve actual value from original
        # feature-store row
        if (
            original_column is not None
            and original_column
            in customer_row.columns
        ):

            original_value = (
                original_row[
                    original_column
                ]
            )

        else:

            original_value = None

        records.append(
            {
                "feature":
                    feature_name,

                "feature_display":
                    clean_feature_name(
                        feature_name
                    ),

                "original_feature":
                    original_column,

                "feature_value":
                    format_feature_value(
                        original_column,
                        original_value
                    ),

                "shap_value":
                    shap_value,

                "abs_shap_value":
                    abs(shap_value),

                "direction":
                    (
                        "Increases churn risk"
                        if shap_value > 0
                        else "Reduces churn risk"
                    ),
            }
        )

    # --------------------------------------------------------
    # SHAP DataFrame
    # --------------------------------------------------------

    shap_df = (
        pd.DataFrame(
            records
        )
        .sort_values(
            "abs_shap_value",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Positive drivers
    # --------------------------------------------------------

    positive = (
        shap_df[
            shap_df[
                "shap_value"
            ] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    # --------------------------------------------------------
    # Negative drivers
    # --------------------------------------------------------

    negative = (
        shap_df[
            shap_df[
                "shap_value"
            ] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(top_n)
        .copy()
    )

    # --------------------------------------------------------
    # Human-readable positive drivers
    # --------------------------------------------------------

    # --------------------------------------------------------
# Human-readable positive drivers
# --------------------------------------------------------

    positive_drivers = []

    for _, row in positive.iterrows():

        positive_drivers.append(
            {
                "feature":
                    row["feature"],

                "display_name":
                    row["feature_display"],

                "value":
                    row["feature_value"],

                "shap_value":
                    float(row["shap_value"]),

                "direction":
                    "Increases churn risk",
            }
        )

    # --------------------------------------------------------
    # Human-readable negative drivers
    # --------------------------------------------------------

# --------------------------------------------------------
# Human-readable negative drivers
# --------------------------------------------------------

    negative_drivers = []

    for _, row in negative.iterrows():

        negative_drivers.append(
            {
                "feature":
                    row["feature"],

                "display_name":
                    row["feature_display"],

                "value":
                    row["feature_value"],

                "shap_value":
                    float(row["shap_value"]),

                "direction":
                    "Reduces churn risk",
            }
        )   
    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "all_shap_values":
            shap_df,

        "positive_drivers":
            positive_drivers,

        "negative_drivers":
            negative_drivers,
    }