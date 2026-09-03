# ============================================================
# RETAIN-AI — PRODUCTION INFERENCE PIPELINE
# ============================================================

from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

CALIBRATED_MODEL_PATH = (
    MODELS_DIR / "xgb_calibrated.joblib"
)

PREPROCESSOR_PATH = (
    MODELS_DIR / "xgb_preprocessor.joblib"
)

BUSINESS_CONFIG_PATH = (
    MODELS_DIR / "business_config.joblib"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the frozen calibrated XGBoost model.
    """

    return joblib.load(
        CALIBRATED_MODEL_PATH
    )


def load_preprocessor():
    """
    Load the frozen preprocessing pipeline.
    """

    return joblib.load(
        PREPROCESSOR_PATH
    )


def load_business_config():
    """
    Load frozen Retain-AI business assumptions.
    """

    return joblib.load(
        BUSINESS_CONFIG_PATH
    )


# ============================================================
# PREPROCESSING
# ============================================================

def transform_features(
    X: pd.DataFrame,
    preprocessor=None
):
    """
    Transform original model features into the
    representation expected by XGBoost.

    Parameters
    ----------
    X : pd.DataFrame
        Original 95 model features.

    preprocessor : optional
        Saved preprocessing pipeline.

    Returns
    -------
    transformed features
    """

    if preprocessor is None:
        preprocessor = load_preprocessor()

    return preprocessor.transform(X)


# ============================================================
# PREDICTION
# ============================================================

def predict_churn_probability(
    X: pd.DataFrame,
    model=None,
    preprocessor=None
):
    """
    Generate calibrated churn probabilities from
    original model-ready features.
    """

    if model is None:
        model = load_model()

    if preprocessor is None:
        preprocessor = load_preprocessor()

    X_transformed = preprocessor.transform(X)

    probabilities = model.predict_proba(
        X_transformed
    )[:, 1]

    return probabilities


# ============================================================
# FULL INFERENCE
# ============================================================

def run_inference(
    X: pd.DataFrame,
    model=None,
    preprocessor=None
):
    """
    Run the complete frozen Retain-AI prediction pipeline.

    Identifier columns such as customer_id are preserved in
    the returned dataframe but are NOT passed to the model.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe containing the model features.
        It may also contain identifier columns.

    model : optional
        Already-loaded calibrated model.

    preprocessor : optional
        Already-loaded preprocessing pipeline.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with churn_probability added.
    """

    if model is None:
        model = load_model()

    if preprocessor is None:
        preprocessor = load_preprocessor()

    result = X.copy()

    # --------------------------------------------------------
    # Identifier / metadata columns
    # --------------------------------------------------------

    metadata_columns = [
        "customer_id",
        "contract_id",
        "snapshot_date"
    ]

    # Keep only columns that actually exist
    metadata_columns = [
        column
        for column in metadata_columns
        if column in result.columns
    ]

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    X_model = result.drop(
        columns=metadata_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    X_transformed = preprocessor.transform(
        X_model
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result["churn_probability"] = (
        model.predict_proba(
            X_transformed
        )[:, 1]
    )

    return result