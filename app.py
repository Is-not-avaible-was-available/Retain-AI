import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Retain-AI — Historical Inference Diagnostic",
    page_icon="🔬",
    layout="wide",
)

st.title("Retain-AI — Historical Inference Diagnostic")

st.write(
    "Testing production XGBoost on a controlled historical batch."
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
# 2. LOAD FEATURES
# ============================================================

st.subheader("2. Load Feature Store")

try:

    start = time.perf_counter()

    features = pd.read_parquet(
        FEATURES_PATH
    )

    features["snapshot_date"] = pd.to_datetime(
        features["snapshot_date"]
    )

    elapsed = time.perf_counter() - start

    st.success(
        f"Feature store loaded in {elapsed:.2f} seconds."
    )

    st.write(
        "Shape:",
        features.shape
    )

    st.write(
        "Memory:",
        f"{features.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )

except Exception as e:

    st.error(
        "Feature store failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 3. PREPARE HISTORICAL DATA
# ============================================================

st.subheader("3. Prepare Historical Dataset")

try:

    historical = (
        features
        .drop(
            columns=[
                "churn_90d"
            ],
            errors="ignore"
        )
        .copy()
        .reset_index(drop=True)
    )

    st.success(
        "Historical dataset prepared."
    )

    st.write(
        "Rows:",
        f"{len(historical):,}"
    )

    st.write(
        "Columns:",
        len(historical.columns)
    )

except Exception as e:

    st.error(
        "Historical dataset preparation failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 4. IMPORT MODEL
# ============================================================

st.subheader("4. Load Production Inference")

try:

    from inference import (
        predict_churn_probability
    )

    st.success(
        "Production inference module imported."
    )

except Exception as e:

    st.error(
        "Inference module import failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 5. CONTROLLED BATCH
# ============================================================

st.subheader(
    "5. Controlled Historical Inference"
)

BATCH_SIZE = 10_000

st.info(
    f"""
Only the first {BATCH_SIZE:,} rows will be scored.

This deliberately avoids scoring the entire
424,607-row feature store.
"""
)

batch = historical.iloc[
    :BATCH_SIZE
].copy()

st.write(
    "Batch shape:",
    batch.shape
)

st.write(
    "Batch snapshot range:",
    str(
        batch[
            "snapshot_date"
        ].min()
    ),
    "→",
    str(
        batch[
            "snapshot_date"
        ].max()
    )
)


# ============================================================
# 6. RUN CONTROLLED INFERENCE
# ============================================================

st.subheader(
    "6. XGBoost Batch Inference"
)

st.warning(
    "Starting production XGBoost inference on 10,000 rows..."
)

start = time.perf_counter()

try:

    probabilities = (
        predict_churn_probability(
            batch
        )
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    st.success(
        "10,000-row XGBoost inference PASSED."
    )

    st.write(
        "Inference time:",
        f"{elapsed:.2f} seconds"
    )

    st.write(
        "Predictions:",
        len(probabilities)
    )

    st.write(
        "Minimum probability:",
        float(
            np.min(probabilities)
        )
    )

    st.write(
        "Maximum probability:",
        float(
            np.max(probabilities)
        )
    )

    st.write(
        "Mean probability:",
        float(
            np.mean(probabilities)
        )
    )

    st.write(
        "NaN predictions:",
        int(
            np.isnan(
                probabilities
            ).sum()
        )
    )

except Exception as e:

    st.error(
        "10,000-row XGBoost inference FAILED."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 7. MEMORY CLEANUP
# ============================================================

del batch
del probabilities

gc.collect()

st.success(
    "Batch objects released and garbage collection completed."
)


# ============================================================
# 8. FINAL
# ============================================================

st.subheader(
    "8. Diagnostic Result"
)

st.success(
    """
The production model successfully processed
a controlled 10,000-row historical batch.

The next test should increase the batch size
rather than immediately scoring the entire
424,607-row feature store.
"""
)