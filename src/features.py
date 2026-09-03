from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROLLING_WINDOWS = [4, 8, 12]


# ============================================================
# Helper functions
# ============================================================

def _safe_pct_change(current, previous):
    """
    Percentage change that safely handles zero denominators.

    If the previous value is zero or extremely close to zero,
    percentage change is undefined and returned as NaN.
    """

    previous = previous.astype(float)
    current = current.astype(float)

    result = np.where(
        previous.abs() > 1e-9,
        (current - previous) / previous.abs(),
        np.nan,
    )

    return result


def _add_lag_change_features(
    df,
    group_col,
    columns,
    lags=(4, 8, 12),
):
    """
    Add lagged absolute and percentage changes.

    The lag is measured in rows within each customer's
    weekly time series.

    Example:
        change_4w = current_week_value - value_4_weeks_ago
    """

    df = df.sort_values(
        [group_col, "week_start"]
    ).copy()

    grouped = df.groupby(
        group_col,
        sort=False,
    )

    for column in columns:

        for lag in lags:

            previous = grouped[column].shift(lag)

            df[f"{column}_change_{lag}w"] = (
                df[column] - previous
            )

            df[f"{column}_pct_change_{lag}w"] = (
                _safe_pct_change(
                    df[column],
                    previous,
                )
            )

    return df


# ============================================================
# Support preparation
# ============================================================

def _prepare_support_weekly(support):
    """
    Convert ticket-level support data into customer-week metrics.

    Week convention:
        W-SUN -> week starts on Monday.
    """

    support = support.copy()

    support["created_at"] = pd.to_datetime(
        support["created_at"]
    )

    support["week_start"] = (
        support["created_at"]
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    support["is_critical"] = (
        support["priority"] == "Critical"
    ).astype(int)

    support["is_high_priority"] = (
        support["priority"] == "High"
    ).astype(int)

    support["is_escalated"] = (
        support["escalated"]
        .astype(bool)
        .astype(int)
    )

    support["is_reopened"] = (
        support["reopened"]
        .astype(bool)
        .astype(int)
    )

    weekly = (
        support
        .groupby(
            [
                "customer_id",
                "week_start",
            ],
            as_index=False,
        )
        .agg(
            ticket_count=(
                "ticket_id",
                "count",
            ),
            avg_csat=(
                "csat_score",
                "mean",
            ),
            avg_resolution_hours=(
                "resolution_hours",
                "mean",
            ),
            escalation_count=(
                "is_escalated",
                "sum",
            ),
            reopen_count=(
                "is_reopened",
                "sum",
            ),
            critical_count=(
                "is_critical",
                "sum",
            ),
            high_priority_count=(
                "is_high_priority",
                "sum",
            ),
        )
    )

    return weekly


def _add_support_rolling_features(
    support_weekly,
    customer_dates,
):
    """
    Build backward-looking support rolling features.

    Important temporal convention:

        support data for week W is assigned to the
        following week's snapshot.

    Therefore, if snapshot_date = Monday 2025-06-02,
    the most recent support week used is the completed
    week beginning 2025-05-26.

    This prevents same-week look-ahead leakage.
    """

    base = customer_dates[
        [
            "customer_id",
            "week_start",
        ]
    ].drop_duplicates()

    support_weekly = base.merge(
        support_weekly,
        on=[
            "customer_id",
            "week_start",
        ],
        how="left",
    )

    count_columns = [
        "ticket_count",
        "escalation_count",
        "reopen_count",
        "critical_count",
        "high_priority_count",
    ]

    for column in count_columns:

        support_weekly[column] = (
            support_weekly[column]
            .fillna(0)
        )

    support_weekly = support_weekly.sort_values(
        [
            "customer_id",
            "week_start",
        ]
    )

    grouped = support_weekly.groupby(
        "customer_id",
        sort=False,
    )

    for window in ROLLING_WINDOWS:

        # ----------------------------------------------------
        # Activity counts
        # ----------------------------------------------------

        support_weekly[
            f"tickets_{window}w"
        ] = (
            grouped["ticket_count"]
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        support_weekly[
            f"escalations_{window}w"
        ] = (
            grouped["escalation_count"]
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        support_weekly[
            f"reopens_{window}w"
        ] = (
            grouped["reopen_count"]
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        support_weekly[
            f"critical_tickets_{window}w"
        ] = (
            grouped["critical_count"]
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        support_weekly[
            f"high_priority_tickets_{window}w"
        ] = (
            grouped["high_priority_count"]
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        # ----------------------------------------------------
        # Quality metrics
        # ----------------------------------------------------

        support_weekly[
            f"avg_csat_{window}w"
        ] = (
            grouped["avg_csat"]
            .rolling(
                window,
                min_periods=1,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        support_weekly[
            f"avg_resolution_hours_{window}w"
        ] = (
            grouped["avg_resolution_hours"]
            .rolling(
                window,
                min_periods=1,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Shift the feature timestamp forward by one week.
    #
    # Data from week 2025-05-26 becomes available for the
    # snapshot on 2025-06-02.
    # --------------------------------------------------------

    support_weekly["snapshot_date"] = (
        support_weekly["week_start"]
        + pd.Timedelta(days=7)
    )

    support_weekly = support_weekly.drop(
        columns=[
            "week_start",
            "ticket_count",
            "avg_csat",
            "avg_resolution_hours",
            "escalation_count",
            "reopen_count",
            "critical_count",
            "high_priority_count",
        ]
    )

    return support_weekly


# ============================================================
# Contract preparation
# ============================================================

def _prepare_contract_features(
    customers,
    contracts,
    snapshots,
):
    """
    Create point-in-time contract features.

    A contract is considered active when:

        start_date <= snapshot_date <= end_date

    Only the contract active at the snapshot is retained.
    """

    contracts = contracts.copy()

    contracts["start_date"] = pd.to_datetime(
        contracts["start_date"]
    )

    contracts["end_date"] = pd.to_datetime(
        contracts["end_date"]
    )

    contracts["renewal_date"] = pd.to_datetime(
        contracts["renewal_date"]
    )

    contract_columns = [
        "customer_id",
        "contract_id",
        "start_date",
        "end_date",
        "renewal_date",
        "annual_contract_value",
        "total_contract_value",
        "contract_type",
        "auto_renew",
        "contract_status",
    ]

    contracts = contracts[
        contract_columns
    ].copy()

    # --------------------------------------------------------
    # Customer × snapshot × contract
    # --------------------------------------------------------

    contract_features = snapshots.merge(
        contracts,
        on="customer_id",
        how="left",
    )

    # --------------------------------------------------------
    # Determine whether contract is active
    # --------------------------------------------------------

    contract_features["contract_active"] = (
        (
            contract_features["snapshot_date"]
            >= contract_features["start_date"]
        )
        &
        (
            contract_features["snapshot_date"]
            <= contract_features["end_date"]
        )
    ).astype(int)

    # --------------------------------------------------------
    # Keep only active-contract attributes
    # --------------------------------------------------------

    inactive_mask = (
        contract_features["contract_active"] == 0
    )

    numeric_contract_columns = [
        "annual_contract_value",
        "total_contract_value",
    ]

    for column in numeric_contract_columns:

        contract_features.loc[
            inactive_mask,
            column,
        ] = np.nan

    contract_features.loc[
        inactive_mask,
        "renewal_date",
    ] = pd.NaT

    contract_features["contract_age_days"] = np.where(
        contract_features["contract_active"] == 1,
        (
            contract_features["snapshot_date"]
            - contract_features["start_date"]
        ).dt.days,
        np.nan,
    )

    contract_features["days_to_renewal"] = np.where(
        contract_features["contract_active"] == 1,
        (
            contract_features["renewal_date"]
            - contract_features["snapshot_date"]
        ).dt.days,
        np.nan,
    )

    contract_features = contract_features.drop(
        columns=[
            "start_date",
            "end_date",
            "renewal_date",
            "contract_status",
        ]
    )

    # --------------------------------------------------------
    # Safety check:
    # each customer-snapshot should have at most one active
    # contract in the current synthetic data.
    # --------------------------------------------------------

    active_contract_counts = (
        contract_features
        .groupby(
            [
                "customer_id",
                "snapshot_date",
            ]
        )["contract_active"]
        .sum()
    )

    assert (
        active_contract_counts.max() <= 1
    ), (
        "Multiple active contracts found for a "
        "customer-snapshot pair."
    )

    return contract_features


# ============================================================
# Product adoption
# ============================================================

def _prepare_product_features(
    subscriptions,
    snapshots,
):
    """
    Create point-in-time product adoption features.

    product_count at snapshot_date is the number of distinct
    products with subscriptions active on that date.

    This prevents future product adoption from leaking into
    earlier snapshots.

    Subscription is considered active when:

        start_date <= snapshot_date <= end_date
    """

    subscriptions = subscriptions.copy()

    subscriptions["start_date"] = pd.to_datetime(
        subscriptions["start_date"]
    )

    subscriptions["end_date"] = pd.to_datetime(
        subscriptions["end_date"]
    )

    subscription_columns = [
        "customer_id",
        "product_id",
        "start_date",
        "end_date",
    ]

    subscriptions = subscriptions[
        subscription_columns
    ].copy()

    # --------------------------------------------------------
    # Customer × snapshot × subscription
    # --------------------------------------------------------

    product_features = snapshots.merge(
        subscriptions,
        on="customer_id",
        how="left",
    )

    # --------------------------------------------------------
    # Determine whether subscription is active
    # --------------------------------------------------------

    product_features["subscription_active"] = (
        (
            product_features["snapshot_date"]
            >= product_features["start_date"]
        )
        &
        (
            product_features["snapshot_date"]
            <= product_features["end_date"]
        )
    )

    # --------------------------------------------------------
    # Count distinct active products
    # --------------------------------------------------------

    active_products = (
        product_features[
            product_features["subscription_active"]
        ]
        .groupby(
            [
                "customer_id",
                "snapshot_date",
            ]
        )["product_id"]
        .nunique()
        .rename("product_count")
        .reset_index()
    )

    # --------------------------------------------------------
    # Merge back so customers with no active product receive 0
    # --------------------------------------------------------

    result = snapshots.merge(
        active_products,
        on=[
            "customer_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    result["product_count"] = (
        result["product_count"]
        .fillna(0)
        .astype(int)
    )

    return result[
        [
            "customer_id",
            "snapshot_date",
            "product_count",
        ]
    ]


# ============================================================
# Main feature generation
# ============================================================

def generate_features(
    customers,
    contracts,
    subscriptions,
    customer_state,
    usage,
    support,
    churn_labels,
):
    """
    Generate a point-in-time customer × snapshot feature table.

    Each row represents a customer at a specific prediction
    snapshot.

    Temporal convention
    -------------------
    snapshot_date = beginning of prediction week.

    Therefore:

    - Customer profile is known as of snapshot.
    - Contract information is known as of snapshot.
    - Product subscriptions are known as of snapshot.
    - Customer state corresponds to snapshot.
    - Usage comes from the previous completed week.
    - Support comes from completed weeks before the snapshot.
    - churn_90d represents future churn after the snapshot.

    This prevents same-week usage/support information from
    leaking into the prediction.
    """

    # ========================================================
    # 1. Prepare target snapshots
    # ========================================================

    labels = churn_labels.copy()

    labels["snapshot_date"] = pd.to_datetime(
        labels["snapshot_date"]
    )

    labels = labels.sort_values(
        [
            "customer_id",
            "snapshot_date",
        ]
    )

    target_snapshots = labels[
        [
            "customer_id",
            "snapshot_date",
            "churn_90d",
        ]
    ].drop_duplicates()

    # ========================================================
    # 2. Customer profile
    # ========================================================

    customer_profile_columns = [
        "customer_id",
        "industry",
        "region",
        "segment",
        "company_size",
        "signup_date",
        "acquisition_channel",
    ]

    customer_profile = customers[
        customer_profile_columns
    ].copy()

    customer_profile["signup_date"] = pd.to_datetime(
        customer_profile["signup_date"]
    )

    features = target_snapshots.merge(
        customer_profile,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # ========================================================
    # 3. Customer tenure
    # ========================================================

    features["tenure_days"] = (
        features["snapshot_date"]
        - features["signup_date"]
    ).dt.days

    features["tenure_years"] = (
        features["tenure_days"]
        / 365.25
    )

    features = features.drop(
        columns=["signup_date"]
    )

    # ========================================================
    # 4. Customer state
    # ========================================================

    state = customer_state.copy()

    state["week_start"] = pd.to_datetime(
        state["week_start"]
    )

    state = state.sort_values(
        [
            "customer_id",
            "week_start",
        ]
    )

    state = _add_lag_change_features(
        state,
        group_col="customer_id",
        columns=[
            "health_score",
        ],
    )

    state_features = state[
        [
            "customer_id",
            "week_start",
            "health_score",
            "baseline_health",
            "health_trend",
            "volatility",
            "behaviour",
            "health_score_change_4w",
            "health_score_change_8w",
            "health_score_change_12w",
            "health_score_pct_change_4w",
            "health_score_pct_change_8w",
            "health_score_pct_change_12w",
        ]
    ].copy()

    state_features = state_features.rename(
        columns={
            "week_start": "snapshot_date"
        }
    )

    features = features.merge(
        state_features,
        on=[
            "customer_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 5. Usage
    # ========================================================

    usage = usage.copy()

    usage["week_start"] = pd.to_datetime(
        usage["week_start"]
    )

    usage = usage.sort_values(
        [
            "customer_id",
            "week_start",
        ]
    )

    usage_change_columns = [
        "active_users",
        "sessions",
        "usage_minutes",
        "active_days",
        "features_used",
    ]

    usage = _add_lag_change_features(
        usage,
        group_col="customer_id",
        columns=usage_change_columns,
    )

    usage_features = usage[
        [
            "customer_id",
            "week_start",
            "active_users",
            "sessions",
            "active_days",
            "usage_minutes",
            "features_used",
            "p001_sessions",
            "p002_sessions",
            "p003_sessions",
            "p004_sessions",
            "p005_sessions",
        ]
        + [
            f"{column}_change_{lag}w"
            for column in usage_change_columns
            for lag in [4, 8, 12]
        ]
        + [
            f"{column}_pct_change_{lag}w"
            for column in usage_change_columns
            for lag in [4, 8, 12]
        ]
    ].copy()

    # --------------------------------------------------------
    # CRITICAL TEMPORAL FIX
    #
    # Usage generated for week W becomes available for the
    # prediction snapshot at W + 1 week.
    #
    # Example:
    #
    # usage week_start = 2025-05-26
    # snapshot_date    = 2025-06-02
    #
    # Thus the model never sees activity occurring during
    # the prediction week.
    # --------------------------------------------------------

    usage_features["snapshot_date"] = (
        usage_features["week_start"]
        + pd.Timedelta(days=7)
    )

    usage_features = usage_features.drop(
        columns=["week_start"]
    )

    features = features.merge(
        usage_features,
        on=[
            "customer_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 6. Usage deterioration indicators
    # ========================================================

    features["usage_decline_4w_flag"] = (
        features["usage_minutes_pct_change_4w"]
        <= -0.15
    ).astype(int)

    features["usage_decline_12w_flag"] = (
        features["usage_minutes_pct_change_12w"]
        <= -0.30
    ).astype(int)

    features["severe_usage_decline_12w_flag"] = (
        features["usage_minutes_pct_change_12w"]
        <= -0.50
    ).astype(int)

    # ========================================================
    # 7. Point-in-time product adoption
    # ========================================================

    product_features = _prepare_product_features(
        subscriptions=subscriptions,
        snapshots=target_snapshots[
            [
                "customer_id",
                "snapshot_date",
            ]
        ],
    )

    features = features.merge(
        product_features,
        on=[
            "customer_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 8. Support weekly + rolling features
    # ========================================================

    support_weekly = _prepare_support_weekly(
        support
    )

    # Use the complete state date range for rolling history.
    support_dates = state[
        [
            "customer_id",
            "week_start",
        ]
    ].drop_duplicates()

    support_features = _add_support_rolling_features(
        support_weekly,
        support_dates,
    )

    support_feature_columns = [
        "customer_id",
        "snapshot_date",
    ]

    for window in ROLLING_WINDOWS:

        support_feature_columns.extend(
            [
                f"tickets_{window}w",
                f"escalations_{window}w",
                f"reopens_{window}w",
                f"critical_tickets_{window}w",
                f"high_priority_tickets_{window}w",
                f"avg_csat_{window}w",
                f"avg_resolution_hours_{window}w",
            ]
        )

    support_features = support_features[
        support_feature_columns
    ]

    features = features.merge(
        support_features,
        on=[
            "customer_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 9. Support-derived indicators
    # ========================================================

    features["high_support_burden_12w"] = (
        features["tickets_12w"] >= 7
    ).astype(int)

    features["high_escalation_12w"] = (
        features["escalations_12w"] >= 2
    ).astype(int)

    features["low_csat_12w"] = (
        features["avg_csat_12w"] <= 3.0
    ).astype(int)

    features["slow_resolution_12w"] = (
        features["avg_resolution_hours_12w"] >= 48
    ).astype(int)

    # ========================================================
    # 10. Contract features
    # ========================================================

    contract_features = _prepare_contract_features(
        customers=customers,
        contracts=contracts,
        snapshots=target_snapshots[
            [
                "customer_id",
                "snapshot_date",
            ]
        ],
    )

    contract_feature_columns = [
        "customer_id",
        "snapshot_date",
        "contract_id",
        "contract_active",
        "annual_contract_value",
        "total_contract_value",
        "contract_type",
        "auto_renew",
        "contract_age_days",
        "days_to_renewal",
    ]

    contract_features = contract_features[
        contract_feature_columns
    ]

    features = features.merge(
        contract_features,
        on=[
            "customer_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 11. Renewal indicators
    # ========================================================

    features["renewal_within_30d"] = (
        features["days_to_renewal"] <= 30
    ).astype(int)

    features["renewal_within_90d"] = (
        features["days_to_renewal"] <= 90
    ).astype(int)

    features["renewal_within_180d"] = (
        features["days_to_renewal"] <= 180
    ).astype(int)

    # ========================================================
    # 12. Final ordering
    # ========================================================

    first_columns = [
        "customer_id",
        "snapshot_date",
        "churn_90d",
    ]

    remaining_columns = [
        column
        for column in features.columns
        if column not in first_columns
    ]

    features = features[
        first_columns
        + remaining_columns
    ]

    features = features.sort_values(
        [
            "customer_id",
            "snapshot_date",
        ]
    ).reset_index(
        drop=True
    )

    return features


# ============================================================
# Validation
# ============================================================

def validate_features(
    features,
    churn_labels,
):
    """
    Validate the structural integrity of the feature table.
    """

    expected_keys = churn_labels[
        [
            "customer_id",
            "snapshot_date",
        ]
    ].drop_duplicates()

    actual_keys = features[
        [
            "customer_id",
            "snapshot_date",
        ]
    ].drop_duplicates()

    expected_keys["snapshot_date"] = pd.to_datetime(
        expected_keys["snapshot_date"]
    )

    actual_keys["snapshot_date"] = pd.to_datetime(
        actual_keys["snapshot_date"]
    )

    expected_keys = expected_keys.sort_values(
        [
            "customer_id",
            "snapshot_date",
        ]
    ).reset_index(
        drop=True
    )

    actual_keys = actual_keys.sort_values(
        [
            "customer_id",
            "snapshot_date",
        ]
    ).reset_index(
        drop=True
    )

    assert len(features) == len(
        expected_keys
    ), (
        "Feature row count does not match labels."
    )

    assert actual_keys.equals(
        expected_keys
    ), (
        "Feature keys do not match target labels."
    )

    assert not features[
        [
            "customer_id",
            "snapshot_date",
        ]
    ].duplicated().any(), (
        "Duplicate customer-snapshot pairs found."
    )

    assert (
        features["churn_90d"].isna().sum() == 0
    ), (
        "Missing target values found."
    )

    # --------------------------------------------------------
    # Additional temporal sanity checks
    # --------------------------------------------------------

    if "tenure_days" in features.columns:

        assert (
            features["tenure_days"] >= 0
        ).all(), (
            "Negative tenure detected."
        )

    if "contract_age_days" in features.columns:

        contract_age = features[
            "contract_age_days"
        ].dropna()

        assert (
            contract_age >= 0
        ).all(), (
            "Negative contract age detected."
        )

    if "days_to_renewal" in features.columns:

        days_to_renewal = features[
            "days_to_renewal"
        ].dropna()

        assert (
            days_to_renewal >= 0
        ).all(), (
            "Negative days-to-renewal detected."
        )

    print(
        "Feature validation passed."
    )

    print(
        f"Rows: {len(features):,}"
    )

    print(
        f"Customers: "
        f"{features['customer_id'].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{features['snapshot_date'].nunique():,}"
    )

    print(
        f"Positive labels: "
        f"{features['churn_90d'].sum():,}"
    )

    print(
        f"Positive rate: "
        f"{features['churn_90d'].mean():.4%}"
    )


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    DATA_DIR = Path("data")

    # --------------------------------------------------------
    # Raw data
    # --------------------------------------------------------

    customers = pd.read_csv(
        DATA_DIR
        / "raw"
        / "customers.csv"
    )

    contracts = pd.read_csv(
        DATA_DIR
        / "raw"
        / "contracts.csv"
    )

    subscriptions = pd.read_csv(
        DATA_DIR
        / "raw"
        / "subscriptions.csv"
    )

    # --------------------------------------------------------
    # Processed data
    # --------------------------------------------------------

    customer_state = pd.read_parquet(
        DATA_DIR
        / "processed"
        / "customer_state.parquet"
    )

    usage = pd.read_parquet(
        DATA_DIR
        / "processed"
        / "usage.parquet"
    )

    support = pd.read_parquet(
        DATA_DIR
        / "processed"
        / "support.parquet"
    )

    churn_labels = pd.read_parquet(
        DATA_DIR
        / "processed"
        / "churn_labels.parquet"
    )

    # --------------------------------------------------------
    # Generate features
    # --------------------------------------------------------

    features = generate_features(
        customers=customers,
        contracts=contracts,
        subscriptions=subscriptions,
        customer_state=customer_state,
        usage=usage,
        support=support,
        churn_labels=churn_labels,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_features(
        features,
        churn_labels,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        DATA_DIR
        / "processed"
        / "features.parquet"
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    sample_path = (
        DATA_DIR
        / "samples"
        / "features_sample.csv"
    )

    features.sample(
        min(
            1000,
            len(features),
        ),
        random_state=42,
    ).to_csv(
        sample_path,
        index=False,
    )

    print(
        f"\nSaved: {output_path}"
    )

    print(
        f"Saved: {sample_path}"
    )