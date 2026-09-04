import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Retain-AI Diagnostic",
    page_icon="🔬",
    layout="wide",
)

st.title("Retain-AI Diagnostic")
st.write("Testing feature-store loading only.")


# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

st.subheader("1. Environment")

st.write("Python version:")
st.code(sys.version)

st.write("NumPy version:", np.__version__)
st.write("Pandas version:", pd.__version__)


# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"

st.write("Feature store path:")
st.code(str(FEATURES_PATH))

st.write("Feature store exists:", FEATURES_PATH.exists())


# ------------------------------------------------------------
# LOAD FEATURE STORE
# ------------------------------------------------------------

st.subheader("2. Loading feature store")

if not FEATURES_PATH.exists():
    st.error("features.parquet was not found.")
    st.stop()

try:

    with st.spinner("Reading features.parquet..."):

        df = pd.read_parquet(
            FEATURES_PATH
        )

    st.success("Feature store loaded successfully.")

    st.write("Shape:")
    st.code(str(df.shape))

    st.write("Columns:")
    st.code(str(len(df.columns)))

    memory_gb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 3)
    )

    st.write(
        f"DataFrame memory usage: {memory_gb:.3f} GB"
    )

    st.write(
        "Unique customers:",
        df["customer_id"].nunique()
    )

    st.write(
        "Snapshot range:",
        str(df["snapshot_date"].min()),
        "→",
        str(df["snapshot_date"].max())
    )

    st.success(
        "Feature-store diagnostic passed."
    )

except Exception as e:

    st.error("Feature-store loading failed.")

    st.exception(e)

    st.stop()


gc.collect()


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

st.subheader("3. Result")

st.success(
    "The feature store can be loaded successfully on Streamlit Cloud."
)

st.info(
    "No model inference or business-layer processing was performed."
)