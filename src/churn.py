import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


# =========================================================
# Churn configuration
# =========================================================

BASE_CHURN_PROBABILITY = {
    "SMB": 0.22,
    "Mid-Market": 0.15,
    "Enterprise": 0.09,
}

BEHAVIOUR_MULTIPLIER = {
    "Stable": 0.60,
    "Improving": 0.45,
    "Declining": 1.70,
    "Rapidly Declining": 3.00,
}

AUTO_RENEW_MULTIPLIER = {
    True: 0.60,
    False: 1.30,
}


# =========================================================
# Health risk
# =========================================================

def _health_risk_multiplier_vectorized(health_score):
    """
    Convert health score into a churn-risk multiplier.

    Lower health -> higher churn risk.
    Higher health -> lower churn risk.
    """

    health_score = np.clip(
        pd.to_numeric(health_score, errors="coerce")
        .fillna(75)
        .to_numpy(dtype=float),
        0,
        100,
    )

    risk = (75 - health_score) / 25

    multiplier = np.exp(
        0.55 * risk
    )

    return np.clip(
        multiplier,
        0.35,
        4.0,
    )


# =========================================================
# Usage risk
# =========================================================

def _prepare_usage_risk(usage):
    """
    Prepare weekly usage deterioration risk.

    Compares:

        recent 6-week average

    against:

        previous 6-week average

    Only information available at the current week is used.
    """

    usage_data = usage[
        [
            "customer_id",
            "week_start",
            "usage_minutes",
        ]
    ].copy()

    if usage_data.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "week_start",
                "usage_risk",
            ]
        )

    usage_data["week_start"] = pd.to_datetime(
        usage_data["week_start"]
    )

    usage_data["usage_minutes"] = pd.to_numeric(
        usage_data["usage_minutes"],
        errors="coerce",
    ).fillna(0.0)

    usage_data = usage_data.sort_values(
        [
            "customer_id",
            "week_start",
        ]
    ).reset_index(drop=True)

    grouped = (
        usage_data
        .groupby("customer_id", sort=False)["usage_minutes"]
    )

    # -----------------------------------------------------
    # Recent 6-week average INCLUDING current week
    # -----------------------------------------------------

    usage_data["recent_usage"] = (
        grouped
        .rolling(
            window=6,
            min_periods=6,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    # -----------------------------------------------------
    # Previous 6-week average
    # -----------------------------------------------------

    usage_data["previous_usage"] = (
        usage_data
        .groupby("customer_id", sort=False)["recent_usage"]
        .shift(6)
    )

    # -----------------------------------------------------
    # Percentage change
    # -----------------------------------------------------

    previous = usage_data[
        "previous_usage"
    ].replace(0, np.nan)

    usage_data["usage_change"] = (
        usage_data["recent_usage"] - previous
    ) / previous

    change = usage_data["usage_change"]

    # -----------------------------------------------------
    # Usage risk multiplier
    # -----------------------------------------------------

    usage_risk = np.ones(
        len(usage_data),
        dtype=float,
    )

    # Severe deterioration
    usage_risk[
        change <= -0.50
    ] = 1.80

    # Strong deterioration
    usage_risk[
        (change > -0.50)
        & (change <= -0.30)
    ] = 1.50

    # Moderate deterioration
    usage_risk[
        (change > -0.30)
        & (change <= -0.15)
    ] = 1.25

    # Strong improvement
    usage_risk[
        change >= 0.30
    ] = 0.85

    # Insufficient history remains 1.0

    usage_data["usage_risk"] = usage_risk

    return usage_data[
        [
            "customer_id",
            "week_start",
            "usage_risk",
        ]
    ]


# =========================================================
# Support risk
# =========================================================

def _prepare_support_risk(support, state_dates):
    """
    Prepare support-risk features for every customer-week.

    Uses the customer's most recent 12 support tickets
    available up to that weekly snapshot.

    Output columns intentionally match the expectations
    of generate_churn().
    """

    dates = state_dates[
        [
            "customer_id",
            "week_start",
        ]
    ].copy()

    dates["week_start"] = pd.to_datetime(
        dates["week_start"]
    )

    # -----------------------------------------------------
    # Empty support safeguard
    # -----------------------------------------------------

    if support.empty:
        return dates.assign(
            support_ticket_count=0.0,
            support_escalation_rate=0.0,
            support_reopen_rate=0.0,
            avg_csat=4.0,
            avg_resolution_hours=24.0,
            support_risk=1.0,
        )

    # -----------------------------------------------------
    # Prepare support data
    # -----------------------------------------------------

    s = support[
        [
            "ticket_id",
            "customer_id",
            "created_at",
            "resolution_hours",
            "reopened",
            "csat_score",
            "escalated",
        ]
    ].copy()

    s["created_at"] = pd.to_datetime(
        s["created_at"]
    )

    s["resolution_hours"] = pd.to_numeric(
        s["resolution_hours"],
        errors="coerce",
    )

    s["csat_score"] = pd.to_numeric(
        s["csat_score"],
        errors="coerce",
    )

    s["resolution_hours"] = (
        s["resolution_hours"]
        .fillna(24.0)
    )

    s["csat_score"] = (
        s["csat_score"]
        .fillna(4.0)
    )

    s["escalated_num"] = (
        s["escalated"]
        .astype(bool)
        .astype(float)
    )

    s["reopened_num"] = (
        s["reopened"]
        .astype(bool)
        .astype(float)
    )

    s = s.sort_values(
        [
            "customer_id",
            "created_at",
        ]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Rolling 12-ticket statistics
    # -----------------------------------------------------

    grouped = s.groupby(
        "customer_id",
        sort=False,
    )

    s["ticket_count"] = (
        grouped["ticket_id"]
        .transform(
            lambda x:
            x.rolling(
                window=12,
                min_periods=1,
            ).count()
        )
    )

    s["escalation_count"] = (
        grouped["escalated_num"]
        .transform(
            lambda x:
            x.rolling(
                window=12,
                min_periods=1,
            ).sum()
        )
    )

    s["reopen_count"] = (
        grouped["reopened_num"]
        .transform(
            lambda x:
            x.rolling(
                window=12,
                min_periods=1,
            ).sum()
        )
    )

    s["csat_sum"] = (
        grouped["csat_score"]
        .transform(
            lambda x:
            x.rolling(
                window=12,
                min_periods=1,
            ).sum()
        )
    )

    s["resolution_sum"] = (
        grouped["resolution_hours"]
        .transform(
            lambda x:
            x.rolling(
                window=12,
                min_periods=1,
            ).sum()
        )
    )

    # -----------------------------------------------------
    # Convert to rates / averages
    # -----------------------------------------------------

    s["support_escalation_rate"] = (
        s["escalation_count"]
        / s["ticket_count"]
    )

    s["support_reopen_rate"] = (
        s["reopen_count"]
        / s["ticket_count"]
    )

    s["avg_csat"] = (
        s["csat_sum"]
        / s["ticket_count"]
    )

    s["avg_resolution_hours"] = (
        s["resolution_sum"]
        / s["ticket_count"]
    )

    # -----------------------------------------------------
    # Prepare as-of support features
    # -----------------------------------------------------

    support_features = s[
        [
            "customer_id",
            "created_at",
            "ticket_count",
            "support_escalation_rate",
            "support_reopen_rate",
            "avg_csat",
            "avg_resolution_hours",
        ]
    ].copy()

    support_features = support_features.rename(
        columns={
            "created_at": "support_date",
            "ticket_count": "support_ticket_count",
        }
    )

    # merge_asof requires sorting by the 'on' column first
    support_features = (
        support_features
        .sort_values(
            [
                "support_date",
                "customer_id",
            ]
        )
        .reset_index(drop=True)
    )

    dates = (
        dates
        .sort_values(
            [
                "week_start",
                "customer_id",
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Match each weekly snapshot with latest support event
    # -----------------------------------------------------

    result = pd.merge_asof(
        dates,
        support_features,
        left_on="week_start",
        right_on="support_date",
        by="customer_id",
        direction="backward",
    )

    # -----------------------------------------------------
    # Fill customers with no support history
    # -----------------------------------------------------

    result["support_ticket_count"] = (
        result["support_ticket_count"]
        .fillna(0.0)
    )

    result["support_escalation_rate"] = (
        result["support_escalation_rate"]
        .fillna(0.0)
    )

    result["support_reopen_rate"] = (
        result["support_reopen_rate"]
        .fillna(0.0)
    )

    result["avg_csat"] = (
        result["avg_csat"]
        .fillna(4.0)
    )

    result["avg_resolution_hours"] = (
        result["avg_resolution_hours"]
        .fillna(24.0)
    )

    # -----------------------------------------------------
    # Support risk multiplier
    # -----------------------------------------------------

    risk = np.ones(
        len(result),
        dtype=float,
    )

    # -----------------------------------------------------
    # Ticket volume
    # -----------------------------------------------------

    risk *= np.where(
        result["support_ticket_count"] >= 10,
        1.30,
        np.where(
            result["support_ticket_count"] >= 7,
            1.18,
            np.where(
                result["support_ticket_count"] >= 4,
                1.08,
                1.00,
            ),
        ),
    )

    # -----------------------------------------------------
    # Escalation
    # -----------------------------------------------------

    risk *= np.where(
        result["support_escalation_rate"] >= 0.30,
        1.20,
        np.where(
            result["support_escalation_rate"] >= 0.15,
            1.10,
            1.00,
        ),
    )

    # -----------------------------------------------------
    # Reopen rate
    # -----------------------------------------------------

    risk *= np.where(
        result["support_reopen_rate"] >= 0.20,
        1.15,
        np.where(
            result["support_reopen_rate"] >= 0.10,
            1.07,
            1.00,
        ),
    )

    # -----------------------------------------------------
    # CSAT
    # -----------------------------------------------------

    risk *= np.where(
        result["avg_csat"] <= 2.5,
        1.30,
        np.where(
            result["avg_csat"] <= 3.0,
            1.20,
            np.where(
                result["avg_csat"] <= 3.5,
                1.10,
                np.where(
                    result["avg_csat"] >= 4.5,
                    0.92,
                    1.00,
                ),
            ),
        ),
    )

    # -----------------------------------------------------
    # Resolution time
    # -----------------------------------------------------

    risk *= np.where(
        result["avg_resolution_hours"] >= 72,
        1.15,
        np.where(
            result["avg_resolution_hours"] >= 48,
            1.08,
            1.00,
        ),
    )

    result["support_risk"] = risk

    # -----------------------------------------------------
    # Return clean output
    # -----------------------------------------------------

    return result[
        [
            "customer_id",
            "week_start",
            "support_ticket_count",
            "support_escalation_rate",
            "support_reopen_rate",
            "avg_csat",
            "avg_resolution_hours",
            "support_risk",
        ]
    ]


# =========================================================
# Churn reason
# =========================================================

def _generate_churn_reason(
    rng,
    health_score,
    behaviour,
    support_risk,
    usage_risk,
    avg_csat,
):
    """
    Generate an interpretable churn reason based on
    the customer's observed risk signals.
    """

    reasons = [
        "Low Product Engagement",
        "Poor Customer Experience",
        "Product / Technical Issues",
        "Business / Budget Constraints",
        "Lack of Required Features",
        "Contract / Renewal Decision",
    ]

    probabilities = np.array(
        [
            0.22,
            0.18,
            0.18,
            0.16,
            0.11,
            0.15,
        ],
        dtype=float,
    )

    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    if health_score < 60:
        probabilities[0] += 0.06
        probabilities[1] += 0.04

    # -----------------------------------------------------
    # Behaviour
    # -----------------------------------------------------

    if behaviour in [
        "Declining",
        "Rapidly Declining",
    ]:
        probabilities[0] += 0.05

    # -----------------------------------------------------
    # Usage
    # -----------------------------------------------------

    if usage_risk >= 1.40:
        probabilities[0] += 0.06

    # -----------------------------------------------------
    # Support
    # -----------------------------------------------------

    if support_risk >= 1.25:
        probabilities[1] += 0.07

    if avg_csat <= 3.2:
        probabilities[1] += 0.07

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    probabilities = np.clip(
        probabilities,
        0.01,
        None,
    )

    probabilities /= probabilities.sum()

    return rng.choice(
        reasons,
        p=probabilities,
    )


# =========================================================
# Main churn generator
# =========================================================

def generate_churn(
    customers: pd.DataFrame,
    contracts: pd.DataFrame,
    customer_state: pd.DataFrame,
    usage: pd.DataFrame,
    support: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate actual customer churn events using a weekly
    hazard-based model.

    Churn probability is influenced by:

        - segment
        - health
        - behaviour
        - usage deterioration
        - support friction
        - auto-renew
        - contract proximity

    A customer can churn at any point during their active
    contract lifecycle.

    Once churn occurs, the customer is removed from all
    future weekly simulations.
    """

    rng = np.random.default_rng(seed)

    # =====================================================
    # Prepare state
    # =====================================================

    state = customer_state[
        [
            "customer_id",
            "week_start",
            "health_score",
            "behaviour",
        ]
    ].copy()

    state["week_start"] = pd.to_datetime(
        state["week_start"]
    )

    # -----------------------------------------------------
    # Add segment from customers
    # -----------------------------------------------------

    customer_segments = customers[
        [
            "customer_id",
            "segment",
        ]
    ].drop_duplicates(
        "customer_id"
    )

    state = state.merge(
        customer_segments,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    missing_segments = state["segment"].isna().sum()

    if missing_segments > 0:
        raise ValueError(
            f"{missing_segments} customer-state rows "
            f"could not be matched to a customer segment."
        )

    state = state.sort_values(
        [
            "week_start",
            "customer_id",
        ]
    ).reset_index(drop=True)

    observation_start = state[
        "week_start"
    ].min()

    observation_end = state[
        "week_start"
    ].max()

    # =====================================================
    # Prepare contracts
    # =====================================================

    contract_data = contracts[
        [
            "contract_id",
            "customer_id",
            "segment",
            "start_date",
            "end_date",
            "auto_renew",
        ]
    ].copy()

    contract_data["start_date"] = pd.to_datetime(
        contract_data["start_date"]
    )

    contract_data["end_date"] = pd.to_datetime(
        contract_data["end_date"]
    )

    # -----------------------------------------------------
    # One primary/latest contract per customer
    # -----------------------------------------------------

    contract_data = (
        contract_data
        .sort_values(
            [
                "customer_id",
                "start_date",
            ]
        )
        .drop_duplicates(
            "customer_id",
            keep="last",
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # Merge contract information
    # =====================================================

    state = state.merge(
        contract_data,
        on="customer_id",
        how="left",
        suffixes=(
            "",
            "_contract",
        ),
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # Validate contract matching
    # -----------------------------------------------------

    missing_contracts = state[
        "contract_id"
    ].isna().sum()

    if missing_contracts > 0:
        raise ValueError(
            f"{missing_contracts} state rows could not "
            f"be matched to a contract."
        )

    # =====================================================
    # Keep only active contract weeks
    # =====================================================

    state = state[
        (state["week_start"] >= state["start_date"])
        & (state["week_start"] <= state["end_date"])
        & (state["week_start"] <= observation_end)
    ].copy()

    # =====================================================
    # Minimum history
    # =====================================================

    minimum_history_date = (
        observation_start
        + pd.Timedelta(weeks=12)
    )

    state = state[
        state["week_start"]
        >= minimum_history_date
    ].copy()

    # =====================================================
    # Usage risk
    # =====================================================

    print("Preparing usage risk...")

    usage_risk = _prepare_usage_risk(
        usage
    )

    state = state.merge(
        usage_risk,
        on=[
            "customer_id",
            "week_start",
        ],
        how="left",
        validate="one_to_one",
    )

    state["usage_risk"] = (
        state["usage_risk"]
        .fillna(1.0)
    )

    # =====================================================
    # Support risk
    # =====================================================

    print("Preparing support risk...")

    state_dates = state[
        [
            "customer_id",
            "week_start",
        ]
    ].copy()

    support_risk = _prepare_support_risk(
        support,
        state_dates,
    )

    state = state.merge(
        support_risk,
        on=[
            "customer_id",
            "week_start",
        ],
        how="left",
        validate="one_to_one",
    )

    state["support_risk"] = (
        state["support_risk"]
        .fillna(1.0)
    )

    state["avg_csat"] = (
        state["avg_csat"]
        .fillna(4.0)
    )

    # =====================================================
    # Health risk
    # =====================================================

    state["health_risk"] = (
        _health_risk_multiplier_vectorized(
            state["health_score"]
        )
    )

    # =====================================================
    # Behaviour risk
    # =====================================================

    state["behaviour_risk"] = (
        state["behaviour"]
        .map(BEHAVIOUR_MULTIPLIER)
        .fillna(1.0)
    )

    # =====================================================
    # Segment base annual probability
    # =====================================================

    state["annual_probability"] = (
        state["segment"]
        .map(BASE_CHURN_PROBABILITY)
    )

    if state["annual_probability"].isna().any():
        unknown_segments = (
            state.loc[
                state["annual_probability"].isna(),
                "segment",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown customer segments found: "
            f"{unknown_segments}"
        )

    # -----------------------------------------------------
    # Convert annual probability into weekly probability
    # -----------------------------------------------------

    state["weekly_base_probability"] = (
        1
        - (
            1
            - state["annual_probability"]
        )
        ** (1 / 52)
    )

    # =====================================================
    # Auto-renew
    # =====================================================

    state["auto_renew_risk"] = (
        state["auto_renew"]
        .map(AUTO_RENEW_MULTIPLIER)
        .fillna(1.0)
    )

    # =====================================================
    # Renewal proximity
    # =====================================================

    state["days_to_contract_end"] = (
        state["end_date"]
        - state["week_start"]
    ).dt.days

    days = state[
        "days_to_contract_end"
    ]

    state["renewal_risk"] = np.where(
        days <= 30,
        1.45,
        np.where(
            days <= 90,
            1.20,
            np.where(
                days <= 180,
                1.08,
                1.00,
            ),
        ),
    )

    # =====================================================
    # Final weekly hazard
    # =====================================================

    state["churn_probability"] = (
        state["weekly_base_probability"]
        * state["behaviour_risk"]
        * state["health_risk"]
        * state["auto_renew_risk"]
        * state["usage_risk"]
        * state["support_risk"]
        * state["renewal_risk"]
    )

    # -----------------------------------------------------
    # Protect against extreme probabilities
    # -----------------------------------------------------

    state["churn_probability"] = np.clip(
        state["churn_probability"],
        0.0005,
        0.12,
    )

    # =====================================================
    # Vectorized weekly churn simulation
    # =====================================================

    print("Simulating weekly churn...")

    state = state.sort_values(
        [
            "week_start",
            "customer_id",
        ]
    ).reset_index(drop=True)

    customer_ids = (
        state["customer_id"]
        .drop_duplicates()
        .to_numpy()
    )

    alive = np.ones(
        len(customer_ids),
        dtype=bool,
    )

    customer_to_position = {
        customer_id: i
        for i, customer_id
        in enumerate(customer_ids)
    }

    churn_rows = []

    # =====================================================
    # Process one week at a time
    # =====================================================

    for current_date, week_data in state.groupby(
        "week_start",
        sort=True,
    ):

        if not alive.any():
            break

        # -------------------------------------------------
        # Map customer IDs to positions
        # -------------------------------------------------

        positions = np.array(
            [
                customer_to_position[cid]
                for cid in week_data["customer_id"]
            ],
            dtype=np.int32,
        )

        currently_alive = alive[
            positions
        ]

        if not currently_alive.any():
            continue

        active_week = week_data.loc[
            currently_alive
        ].copy()

        active_positions = positions[
            currently_alive
        ]

        # -------------------------------------------------
        # Random churn draw
        # -------------------------------------------------

        probabilities = (
            active_week[
                "churn_probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        random_values = rng.random(
            len(active_week)
        )

        churn_mask = (
            random_values
            < probabilities
        )

        if not churn_mask.any():
            continue

        churned_week = active_week.loc[
            churn_mask
        ].copy()

        churned_positions = (
            active_positions[
                churn_mask
            ]
        )

        # -------------------------------------------------
        # Generate churn day within current week
        # -------------------------------------------------

        timing_offsets = rng.integers(
            0,
            7,
            size=len(churned_week),
        )

        churn_dates = (
            pd.to_datetime(
                churned_week[
                    "week_start"
                ]
            )
            + pd.to_timedelta(
                timing_offsets,
                unit="D",
            )
        )

        # Do not allow churn after observation period.
        churn_dates = pd.Series(
            churn_dates,
            index=churned_week.index,
        ).clip(
            upper=observation_end
        )

        # -------------------------------------------------
        # Generate reasons
        # -------------------------------------------------

        for i, (_, row) in enumerate(
            churned_week.iterrows()
        ):

            reason = _generate_churn_reason(
                rng=rng,
                health_score=float(
                    row["health_score"]
                ),
                behaviour=row["behaviour"],
                support_risk=float(
                    row["support_risk"]
                ),
                usage_risk=float(
                    row["usage_risk"]
                ),
                avg_csat=float(
                    row["avg_csat"]
                ),
            )

            churn_rows.append(
                {
                    "customer_id": row[
                        "customer_id"
                    ],

                    "contract_id": row[
                        "contract_id"
                    ],

                    "churn_date": churn_dates.iloc[
                        i
                    ],

                    "churn_reason": reason,

                    "churn_probability": round(
                        float(
                            row[
                                "churn_probability"
                            ]
                        ),
                        4,
                    ),

                    "health_score_at_churn": round(
                        float(
                            row[
                                "health_score"
                            ]
                        ),
                        2,
                    ),

                    "behaviour_at_churn": row[
                        "behaviour"
                    ],
                }
            )

        # -------------------------------------------------
        # Remove churned customers from future weeks
        # -------------------------------------------------

        alive[
            churned_positions
        ] = False

    # =====================================================
    # Output
    # =====================================================

    columns = [
        "customer_id",
        "contract_id",
        "churn_date",
        "churn_reason",
        "churn_probability",
        "health_score_at_churn",
        "behaviour_at_churn",
    ]

    if not churn_rows:
        return pd.DataFrame(
            columns=columns
        )

    churn = pd.DataFrame(
        churn_rows,
        columns=columns,
    )

    churn["churn_date"] = pd.to_datetime(
        churn["churn_date"]
    )

    # -----------------------------------------------------
    # One churn event per customer
    # -----------------------------------------------------

    churn = (
        churn
        .sort_values(
            [
                "churn_date",
                "customer_id",
            ]
        )
        .drop_duplicates(
            "customer_id",
            keep="first",
        )
        .reset_index(drop=True)
    )

    return churn