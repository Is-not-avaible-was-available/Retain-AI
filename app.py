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
    page_title="Retain-AI — Memory Diagnostic",
    page_icon="🔬",
    layout="wide",
)

st.title("Retain-AI — Memory / Historical Inference Diagnostic")

st.write(
    "Testing whether the historical inference failure is caused "
    "by duplicated feature-store memory."
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
# 2. LOAD FEATURE STORE
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

    elapsed = (
        time.perf_counter()
        -
        start
    )

    memory_mb = (
        features.memory_usage(
            deep=True
        ).sum()
        /
        1024**2
    )

    st.success(
        f"Feature store loaded in {elapsed:.2f} seconds."
    )

    st.write(
        "Shape:",
        features.shape
    )

    st.write(
        "Pandas memory:",
        f"{memory_mb:.1f} MB"
    )

except Exception as e:

    st.error(
        "Feature store loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 3. EXTRACT SMALL BATCH
# ============================================================

st.subheader(
    "3. Extract 10,000-Row Batch"
)

try:

    BATCH_SIZE = 10_000

    batch = (
        features
        .iloc[:BATCH_SIZE]
        .copy()
        .reset_index(drop=True)
    )

    st.success(
        "10,000-row batch created."
    )

    st.write(
        "Batch shape:",
        batch.shape
    )

    st.write(
        "Batch memory:",
        f"{batch.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )

except Exception as e:

    st.error(
        "Batch creation failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 4. RELEASE FULL FEATURE STORE
# ============================================================

st.subheader(
    "4. Release Full Feature Store"
)

try:

    del features

    gc.collect()

    st.success(
        """
Full 424,607-row feature store released from the
Python process. Only the 10,000-row batch remains.
"""
    )

except Exception as e:

    st.error(
        "Memory cleanup failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 5. IMPORT PRODUCTION INFERENCE
# ============================================================

st.subheader(
    "5. Load Production Inference"
)

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
# 6. RUN XGBOOST
# ============================================================

st.subheader(
    "6. XGBoost Inference — Memory-Controlled Test"
)

st.warning(
    """
Only 10,000 rows are being scored, and the original
424,607-row feature store has already been deleted.
"""
)

st.write(
    "Starting inference..."
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
        "XGBoost inference PASSED."
    )

    st.write(
        "Inference time:",
        f"{elapsed:.2f} seconds"
    )

    st.write(
        "Prediction count:",
        len(probabilities)
    )

    st.write(
        "Minimum probability:",
        float(
            np.min(
                probabilities
            )
        )
    )

    st.write(
        "Maximum probability:",
        float(
            np.max(
                probabilities
            )
        )
    )

    st.write(
        "Mean probability:",
        float(
            np.mean(
                probabilities
            )
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
        "XGBoost inference FAILED."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 7. CLEANUP
# ============================================================

st.subheader(
    "7. Final Memory Cleanup"
)

del batch
del probabilities

gc.collect()

st.success(
    "Batch and predictions released successfully."
)


# ============================================================
# 8. FINAL RESULT
# ============================================================

st.subheader(
    "8. Diagnostic Result"
)

st.success(
    """
MEMORY-CONTROLLED HISTORICAL INFERENCE TEST PASSED.

The production XGBoost model can score a historical
batch when the full feature store is not simultaneously
held in memory.
"""
)

st.write(
    """
If this test passes, the production application should
NOT perform historical inference by creating another
full copy of the feature store.

The correct production fix will be to process historical
snapshots in memory-controlled batches.
"""
)