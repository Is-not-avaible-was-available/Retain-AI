import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


# =========================================================
# Usage configuration
# =========================================================

# Approximate sessions per active user per week
PRODUCT_SESSION_RATE = {
    "P001": 4.0,   # Core Platform
    "P002": 2.5,   # Analytics
    "P003": 2.8,   # Workflow Automation
    "P004": 2.2,   # AI Assistant
    "P005": 1.8,   # Advanced Reporting
}


# Probability that an active user engages with a
# subscribed product during a given week.
PRODUCT_USAGE_PROBABILITY = {
    "P001": 0.95,
    "P002": 0.75,
    "P003": 0.70,
    "P004": 0.65,
    "P005": 0.60,
}


# Approximate session duration in minutes.
PRODUCT_SESSION_DURATION = {
    "P001": 18,
    "P002": 22,
    "P003": 20,
    "P004": 15,
    "P005": 24,
}


# =========================================================
# Segment-specific usage assumptions
# =========================================================

# Approximate percentage of employees who could become
# active platform users.
USER_ADOPTION_RATE = {
    "SMB": (0.10, 0.30),
    "Mid-Market": (0.08, 0.20),
    "Enterprise": (0.05, 0.15),
}


# =========================================================
# Customer baseline generation
# =========================================================

def _generate_customer_baselines(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:

    baseline = customers[
        [
            "customer_id",
            "segment",
            "company_size",
        ]
    ].copy()

    # -----------------------------------------------------
    # Customer-specific adoption rate
    # -----------------------------------------------------

    low_rates = baseline["segment"].map(
        lambda segment: USER_ADOPTION_RATE[segment][0]
    )

    high_rates = baseline["segment"].map(
        lambda segment: USER_ADOPTION_RATE[segment][1]
    )

    baseline["user_adoption_rate"] = (
        rng.uniform(
            low_rates.to_numpy(),
            high_rates.to_numpy(),
        )
    )

    # -----------------------------------------------------
    # Baseline active users
    # -----------------------------------------------------

    baseline["baseline_active_users"] = np.maximum(
        1,
        np.round(
            baseline["company_size"]
            * baseline["user_adoption_rate"]
        ).astype(int),
    )

    # -----------------------------------------------------
    # Persistent customer-specific engagement effect
    #
    # Two customers with the same company size can still
    # have different natural engagement levels.
    # -----------------------------------------------------

    baseline["engagement_multiplier"] = (
        rng.lognormal(
            mean=0,
            sigma=0.20,
            size=len(baseline),
        )
    )

    baseline["baseline_active_users"] = np.maximum(
        1,
        np.round(
            baseline["baseline_active_users"]
            * baseline["engagement_multiplier"]
        ).astype(int),
    )

    return baseline


# =========================================================
# Main usage generator
# =========================================================

def generate_usage(
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    customer_state: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # Customer baselines
    # -----------------------------------------------------

    baseline = _generate_customer_baselines(
        customers,
        rng,
    )

    # -----------------------------------------------------
    # Prepare subscription data
    # -----------------------------------------------------

    subscription_data = subscriptions[
        [
            "customer_id",
            "product_id",
            "start_date",
            "end_date",
        ]
    ].copy()

    subscription_data["start_date"] = pd.to_datetime(
        subscription_data["start_date"]
    )

    subscription_data["end_date"] = pd.to_datetime(
        subscription_data["end_date"]
    )

    # -----------------------------------------------------
    # Merge customer state with baseline attributes
    # -----------------------------------------------------

    usage = customer_state.merge(
        baseline,
        on="customer_id",
        how="left",
    )

    # -----------------------------------------------------
    # Customer-level contract window
    # -----------------------------------------------------

    contract_dates = (
        subscription_data
        .groupby("customer_id")
        .agg(
            customer_start=("start_date", "min"),
            customer_end=("end_date", "max"),
        )
        .reset_index()
    )

    usage = usage.merge(
        contract_dates,
        on="customer_id",
        how="left",
    )

    # -----------------------------------------------------
    # Determine whether customer is contract-active
    # -----------------------------------------------------

    usage["is_active"] = (
        usage["week_start"] >= usage["customer_start"]
    ) & (
        usage["week_start"] <= usage["customer_end"]
    )

    # -----------------------------------------------------
    # Number of subscribed products
    # -----------------------------------------------------

    product_counts = (
        subscription_data
        .groupby("customer_id")["product_id"]
        .nunique()
        .rename("product_count")
        .reset_index()
    )

    usage = usage.merge(
        product_counts,
        on="customer_id",
        how="left",
    )

    usage["product_count"] = (
        usage["product_count"]
        .fillna(0)
        .astype(int)
    )

    # =====================================================
    # Health-driven engagement
    # =====================================================

    # Health score of 75 represents approximately baseline
    # engagement.
    #
    # Higher health → higher usage
    # Lower health  → lower usage

    health_ratio = (
        usage["health_score"] / 75
    )

    usage["health_factor"] = np.clip(
        health_ratio ** 2.0,
        0.25,
        1.60,
    )

    # =====================================================
    # Product multiplier
    # =====================================================

    usage["product_factor"] = (
        1
        + 0.15
        * np.maximum(
            usage["product_count"] - 1,
            0,
        )
    )

    # =====================================================
    # Seasonality
    # =====================================================

    week_of_year = (
        usage["week_start"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    usage["seasonality_factor"] = (
        1
        + 0.03
        * np.sin(
            2
            * np.pi
            * week_of_year
            / 52,
        )
    )

    # =====================================================
    # Target active users
    # =====================================================

    usage["target_active_users"] = (
        usage["baseline_active_users"]
        * usage["health_factor"]
        * usage["product_factor"]
        * usage["seasonality_factor"]
    )

    # =====================================================
    # Persistent active-user generation
    # =====================================================

    usage = usage.sort_values(
        [
            "customer_id",
            "week_start",
        ]
    ).reset_index(drop=True)

    active_users = np.zeros(
        len(usage),
        dtype=int,
    )

    for customer_id, indices in usage.groupby(
        "customer_id"
    ).groups.items():

        indices = list(indices)

        previous_users = None

        for idx in indices:

            # ---------------------------------------------
            # Outside contract period
            # ---------------------------------------------

            if not usage.loc[idx, "is_active"]:

                active_users[idx] = 0
                previous_users = None

                continue

            target = usage.loc[
                idx,
                "target_active_users",
            ]

            # ---------------------------------------------
            # First active week
            # ---------------------------------------------

            if previous_users is None:

                current_users = (
                    target
                    * rng.lognormal(
                        mean=0,
                        sigma=0.05,
                    )
                )

            # ---------------------------------------------
            # Subsequent weeks
            # ---------------------------------------------

            else:

                # Usage has memory.
                #
                # 60% comes from previous engagement.
                # 40% responds to the current target,
                # which is influenced by health.

                current_users = (
                    0.60 * previous_users
                    + 0.40 * target
                )

                # Small week-to-week variation.
                current_users *= rng.lognormal(
                    mean=0,
                    sigma=0.025,
                )

            current_users = max(
                1,
                round(current_users),
            )

            active_users[idx] = current_users

            previous_users = current_users

    usage["active_users"] = np.where(
        usage["is_active"],
        active_users,
        0,
    )

    # =====================================================
    # Product-level sessions
    # =====================================================

    product_ids = [
        "P001",
        "P002",
        "P003",
        "P004",
        "P005",
    ]

    for product_id in product_ids:

        session_column = (
            f"{product_id.lower()}_sessions"
        )

        # -------------------------------------------------
        # Customers subscribed to this product
        # -------------------------------------------------

        product_subscribers = set(
            subscription_data.loc[
                subscription_data["product_id"]
                == product_id,
                "customer_id",
            ]
        )

        owns_product = (
            usage["customer_id"]
            .isin(product_subscribers)
        )

        # -------------------------------------------------
        # Product engagement probability
        # -------------------------------------------------

        base_probability = (
            PRODUCT_USAGE_PROBABILITY[
                product_id
            ]
        )

        effective_probability = (
            base_probability
            * (
                0.60
                + 0.40
                * (
                    usage["health_score"]
                    / 100
                )
            )
        )

        effective_probability = np.clip(
            effective_probability,
            0,
            1,
        )

        # -------------------------------------------------
        # Number of active users engaging with product
        # -------------------------------------------------

        engaging_users = rng.binomial(
            n=np.maximum(
                usage["active_users"],
                0,
            ),
            p=effective_probability,
        )

        # -------------------------------------------------
        # Generate sessions
        # -------------------------------------------------

        sessions = rng.poisson(
            np.maximum(
                engaging_users
                * PRODUCT_SESSION_RATE[
                    product_id
                ],
                0,
            )
        )

        usage[session_column] = np.where(
            usage["is_active"]
            & owns_product,
            sessions,
            0,
        )

    # =====================================================
    # Total sessions
    # =====================================================

    session_columns = [
        f"{product_id.lower()}_sessions"
        for product_id in product_ids
    ]

    usage["sessions"] = (
        usage[session_columns]
        .sum(axis=1)
    )

    # =====================================================
    # Active days
    # =====================================================

    session_intensity = np.minimum(
        usage["sessions"]
        / (
            usage["active_users"] * 5
            + 1
        ),
        1,
    )

    active_day_probability = np.clip(
        0.25
        + 0.65 * session_intensity,
        0,
        0.95,
    )

    active_days = rng.binomial(
        n=7,
        p=active_day_probability,
    )

    usage["active_days"] = np.where(
        usage["is_active"],
        active_days,
        0,
    )

    # =====================================================
    # Usage minutes
    # =====================================================

    usage_minutes = np.zeros(
        len(usage)
    )

    for product_id in product_ids:

        session_column = (
            f"{product_id.lower()}_sessions"
        )

        sessions = (
            usage[session_column]
            .to_numpy()
        )

        duration = PRODUCT_SESSION_DURATION[
            product_id
        ]

        # Session duration varies from week to week.
        duration_noise = rng.lognormal(
            mean=0,
            sigma=0.12,
            size=len(usage),
        )

        usage_minutes += (
            sessions
            * duration
            * duration_noise
        )

    usage["usage_minutes"] = (
        np.round(
            usage_minutes,
            0,
        ).astype(int)
    )

    # =====================================================
    # Features used
    # =====================================================

    usage["features_used"] = (
        usage[session_columns]
        .gt(0)
        .sum(axis=1)
    )

    # =====================================================
    # Remove internal/helper columns
    # =====================================================

    usage = usage.drop(
        columns=[
            "customer_start",
            "customer_end",
            "user_adoption_rate",
            "engagement_multiplier",
            "baseline_active_users",
            "health_factor",
            "product_factor",
            "seasonality_factor",
            "target_active_users",
            "is_active",
        ]
    )

    # =====================================================
    # Final column ordering
    # =====================================================

    usage = usage[
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
    ]

    return usage