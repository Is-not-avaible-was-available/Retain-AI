# src/labels.py

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


PREDICTION_HORIZON_DAYS = 90


def generate_churn_labels(
    customers,
    churn,
    customer_state,
    contracts,
    observation_end=None,
    horizon_days=PREDICTION_HORIZON_DAYS,
    seed=RANDOM_SEED,
):
    """
    Generate leakage-safe weekly 90-day churn labels.

    Each row represents an eligible customer at a historical
    weekly snapshot date.

    churn_90d = 1 when:
        snapshot_date < churn_date <= snapshot_date + 90 days

    churn_90d = 0 when:
        the customer does not churn within the next 90 days.

    A snapshot is eligible only when:

        signup_date <= snapshot_date
        AND
        contract_start <= snapshot_date <= contract_end

    Important:
    - Only active customer lifecycle periods are included.
    - Snapshots after actual churn are removed.
    - The churn event itself is never included as a feature.
    - The final 90-day period is excluded because its complete
      outcome is not observable inside the observation window.
    """

    rng = np.random.default_rng(seed)

    # =========================================================
    # PREPARE DATA
    # =========================================================

    state = customer_state.copy()

    state["week_start"] = pd.to_datetime(
        state["week_start"]
    )

    churn_data = churn.copy()

    churn_data["churn_date"] = pd.to_datetime(
        churn_data["churn_date"]
    )

    customer_data = customers.copy()

    customer_data["signup_date"] = pd.to_datetime(
        customer_data["signup_date"]
    )

    contract_data = contracts.copy()

    contract_data["start_date"] = pd.to_datetime(
        contract_data["start_date"]
    )

    contract_data["end_date"] = pd.to_datetime(
        contract_data["end_date"]
    )

    if observation_end is None:
        observation_end = state["week_start"].max()
    else:
        observation_end = pd.Timestamp(
            observation_end
        )

    horizon = pd.Timedelta(
        days=horizon_days
    )

    # =========================================================
    # ELIGIBLE SNAPSHOT DATES
    #
    # We only create snapshots where the entire 90-day
    # prediction horizon exists inside the observation period.
    # =========================================================

    latest_snapshot_date = (
        observation_end - horizon
    )

    snapshot_dates = (
        state.loc[
            state["week_start"] <= latest_snapshot_date,
            "week_start",
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    # =========================================================
    # CUSTOMER × SNAPSHOT DATE
    # =========================================================

    customer_ids = (
        customer_data["customer_id"]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    snapshots = pd.MultiIndex.from_product(
        [
            customer_ids,
            snapshot_dates,
        ],
        names=[
            "customer_id",
            "snapshot_date",
        ],
    ).to_frame(index=False)

    # =========================================================
    # CUSTOMER SIGNUP ELIGIBILITY
    #
    # A customer cannot be predicted as a churn risk before
    # they have actually signed up.
    # =========================================================

    signup_dates = customer_data[
        [
            "customer_id",
            "signup_date",
        ]
    ].drop_duplicates(
        "customer_id"
    )

    snapshots = snapshots.merge(
        signup_dates,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    snapshots = snapshots[
        snapshots["snapshot_date"]
        >= snapshots["signup_date"]
    ].copy()

    snapshots = snapshots.drop(
        columns=["signup_date"]
    )

    # =========================================================
    # STATE ELIGIBILITY
    #
    # Only retain customer × snapshot pairs for which the
    # customer actually has a state observation.
    # =========================================================

    state_dates = (
        state[
            [
                "customer_id",
                "week_start",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "week_start": "snapshot_date"
            }
        )
    )

    snapshots = snapshots.merge(
        state_dates.assign(
            has_state=True
        ),
        on=[
            "customer_id",
            "snapshot_date",
        ],
        how="left",
    )

    snapshots = snapshots[
        snapshots["has_state"].eq(True)
    ].drop(
        columns=["has_state"]
    )

    # =========================================================
    # CONTRACT LIFECYCLE ELIGIBILITY
    #
    # Only retain snapshots where the customer has an active
    # contract.
    #
    # start_date <= snapshot_date <= end_date
    # =========================================================

    contract_lifecycle = contract_data[
        [
            "customer_id",
            "contract_id",
            "start_date",
            "end_date",
        ]
    ].copy()

    snapshots = snapshots.merge(
        contract_lifecycle,
        on="customer_id",
        how="left",
        validate="many_to_many",
    )

    snapshots["contract_active"] = (
        (
            snapshots["snapshot_date"]
            >= snapshots["start_date"]
        )
        &
        (
            snapshots["snapshot_date"]
            <= snapshots["end_date"]
        )
    )

    snapshots = snapshots[
        snapshots["contract_active"]
    ].copy()

    # Keep the contract ID temporarily so we can ensure that
    # every prediction row belongs to a real contract.
    snapshots = snapshots[
        [
            "customer_id",
            "snapshot_date",
            "contract_id",
        ]
    ]

    # =========================================================
    # REMOVE DUPLICATE CUSTOMER × SNAPSHOT PAIRS
    #
    # Normally unnecessary with the current one-contract-per-
    # customer setup, but protects the pipeline if contracts
    # later contain overlapping records.
    # =========================================================

    snapshots = (
        snapshots
        .sort_values(
            [
                "customer_id",
                "snapshot_date",
            ]
        )
        .drop_duplicates(
            [
                "customer_id",
                "snapshot_date",
            ]
        )
    )

    # =========================================================
    # MERGE ACTUAL CHURN DATE
    # =========================================================

    churn_events = (
        churn_data[
            [
                "customer_id",
                "churn_date",
            ]
        ]
        .drop_duplicates(
            "customer_id"
        )
    )

    snapshots = snapshots.merge(
        churn_events,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # =========================================================
    # REMOVE POST-CHURN SNAPSHOTS
    #
    # Once a customer has actually churned, they should no
    # longer appear in the prediction population.
    # =========================================================

    snapshots = snapshots[
        snapshots["churn_date"].isna()
        |
        (
            snapshots["snapshot_date"]
            < snapshots["churn_date"]
        )
    ].copy()

    # =========================================================
    # CREATE 90-DAY CHURN LABEL
    #
    # snapshot_date < churn_date <= snapshot_date + 90 days
    # =========================================================

    snapshots["churn_90d"] = (
        snapshots["churn_date"].notna()
        &
        (
            snapshots["churn_date"]
            > snapshots["snapshot_date"]
        )
        &
        (
            snapshots["churn_date"]
            <= snapshots["snapshot_date"]
            + horizon
        )
    ).astype("int8")

    # =========================================================
    # TEMPORARY VALIDATION COLUMN
    # =========================================================

    snapshots["days_to_churn"] = np.where(
        snapshots["churn_date"].notna(),
        (
            snapshots["churn_date"]
            - snapshots["snapshot_date"]
        ).dt.days,
        np.nan,
    )

    # =========================================================
    # FINAL LABEL TABLE
    #
    # Contract ID, churn date, and days-to-churn are deliberately
    # removed.
    #
    # The ML feature pipeline must not use the actual churn event.
    # =========================================================

    labels = snapshots[
        [
            "customer_id",
            "snapshot_date",
            "churn_90d",
        ]
    ].copy()

    # =========================================================
    # SORT
    # =========================================================

    labels = (
        labels
        .sort_values(
            [
                "customer_id",
                "snapshot_date",
            ]
        )
        .reset_index(drop=True)
    )

    return labels