import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Retain-AI Diagnostic",
    page_icon="🔬",
    layout="wide",
)

st.title("Retain-AI — Model Inference Diagnostic")

st.write("Testing feature store + portfolio preparation + XGBoost inference.")


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


# ============================================================
# ENVIRONMENT
# ============================================================

st.subheader("1. Environment")

st.write("Python:")
st.code(sys.version)

st.write("NumPy:", np.__version__)
st.write("Pandas:", pd.__version__)


# ============================================================
# LOAD FEATURES
# ============================================================

st.subheader("2. Loading feature store")

try:

    with st.spinner("Loading feature store..."):

        features = pd.read_parquet(
            FEATURES_PATH
        )

    features["snapshot_date"] = pd.to_datetime(
        features["snapshot_date"]
    )

    st.success("Feature store loaded.")

    st.write("Shape:", features.shape)

except Exception as e:

    st.error("Feature-store loading failed.")
    st.exception(e)
    st.stop()


# ============================================================
# PREPARE PORTFOLIO
# ============================================================

st.subheader("3. Preparing latest portfolio")

try:

    latest_date = features["snapshot_date"].max()

    portfolio = (
        features[
            features["snapshot_date"] == latest_date
        ]
        .copy()
        .reset_index(drop=True)
    )

    portfolio = portfolio.drop(
        columns=["churn_90d"],
        errors="ignore",
    )

    st.success("Portfolio prepared.")

    st.write("Latest snapshot:", latest_date)
    st.write("Portfolio shape:", portfolio.shape)
    st.write(
        "Unique customers:",
        portfolio["customer_id"].nunique(),
    )

except Exception as e:

    st.error("Portfolio preparation failed.")
    st.exception(e)
    st.stop()


# ============================================================
# MODEL INFERENCE
# ============================================================

st.subheader("4. Running XGBoost inference")

try:

    # Make src available
    SRC_DIR = PROJECT_ROOT / "src"

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from inference import predict_churn_probability

    st.write("Inference module imported successfully.")

    with st.spinner(
        "Running calibrated XGBoost inference..."
    ):

        probabilities = predict_churn_probability(
            portfolio
        )

    st.success("XGBoost inference completed.")

    st.write(
        "Number of predictions:",
        len(probabilities),
    )

    st.write(
        "Minimum probability:",
        float(np.min(probabilities)),
    )

    st.write(
        "Maximum probability:",
        float(np.max(probabilities)),
    )

    st.write(
        "Mean probability:",
        float(np.mean(probabilities)),
    )

    st.write(
        "NaN predictions:",
        int(np.isnan(probabilities).sum()),
    )

    st.success(
        "MODEL INFERENCE DIAGNOSTIC PASSED."
    )

except Exception as e:

    st.error(
        "XGBoost inference failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# COMPLETE
# ============================================================

st.subheader("5. Result")

st.success(
    """
Feature loading, portfolio preparation, and
calibrated XGBoost inference all work successfully
on Streamlit Cloud.
"""
)