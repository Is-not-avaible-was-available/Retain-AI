import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


# =========================================================
# Support configuration
# =========================================================

ISSUE_CATEGORIES = [
    "Technical Issue",
    "Product Bug",
    "Integration",
    "Performance",
    "Billing",
    "Feature Request",
    "Account Management",
    "Training / Onboarding",
]


CHANNELS = [
    "Email",
    "Chat",
    "Phone",
    "Support Portal",
]


# Base probability of each issue category.
ISSUE_CATEGORY_PROBABILITY = {
    "Technical Issue": 0.24,
    "Product Bug": 0.16,
    "Integration": 0.13,
    "Performance": 0.12,
    "Billing": 0.10,
    "Feature Request": 0.10,
    "Account Management": 0.08,
    "Training / Onboarding": 0.07,
}


# Base probability of each support channel.
CHANNEL_PROBABILITY = {
    "Email": 0.30,
    "Chat": 0.30,
    "Phone": 0.20,
    "Support Portal": 0.20,
}


# =========================================================
# Segment-specific support behaviour
# =========================================================

# Expected support interactions per customer per month
# at approximately average health.
BASE_TICKETS_PER_MONTH = {
    "SMB": 0.45,
    "Mid-Market": 0.85,
    "Enterprise": 1.50,
}


# Base priority distribution by customer segment.
#
# Enterprise customers are assumed to have more
# complex/high-impact support requirements.
PRIORITY_WEIGHTS = {
    "SMB": {
        "Low": 0.30,
        "Medium": 0.50,
        "High": 0.17,
        "Critical": 0.03,
    },
    "Mid-Market": {
        "Low": 0.20,
        "Medium": 0.48,
        "High": 0.25,
        "Critical": 0.07,
    },
    "Enterprise": {
        "Low": 0.15,
        "Medium": 0.43,
        "High": 0.32,
        "Critical": 0.10,
    },
}


# =========================================================
# Helpers
# =========================================================

def _choose_weighted(
    rng: np.random.Generator,
    values,
    probabilities,
    size=None,
):
    """
    Choose one or more values according to supplied
    probability weights.
    """

    return rng.choice(
        values,
        size=size,
        p=probabilities,
    )


def _get_issue_probabilities(
    health_score: float,
) -> dict:
    """
    Adjust issue probabilities based on customer health.

    Lower-health customers are more likely to experience
    technical, product, and performance-related issues.

    Health is used only during generation and is NOT stored
    in the final support dataset.
    """

    probabilities = ISSUE_CATEGORY_PROBABILITY.copy()

    if health_score < 60:

        probabilities["Technical Issue"] += 0.06
        probabilities["Product Bug"] += 0.04
        probabilities["Performance"] += 0.05

        probabilities["Feature Request"] -= 0.04
        probabilities["Training / Onboarding"] -= 0.03

    elif health_score > 85:

        probabilities["Feature Request"] += 0.03
        probabilities["Training / Onboarding"] += 0.02

        probabilities["Technical Issue"] -= 0.02
        probabilities["Performance"] -= 0.02

    # Prevent negative probabilities.
    probabilities = {
        key: max(value, 0)
        for key, value in probabilities.items()
    }

    # Normalize so probabilities sum to 1.
    total = sum(probabilities.values())

    return {
        key: value / total
        for key, value in probabilities.items()
    }


def _generate_resolution_hours(
    rng: np.random.Generator,
    priority: str,
    issue_category: str,
) -> float:
    """
    Generate a realistic resolution time.

    Higher-priority issues are generally resolved faster,
    while complex issues such as integrations take longer.

    Lognormal noise prevents resolution times from looking
    artificially uniform.
    """

    priority_base = {
        "Low": 48,
        "Medium": 24,
        "High": 12,
        "Critical": 4,
    }

    issue_multiplier = {
        "Technical Issue": 1.20,
        "Product Bug": 1.50,
        "Integration": 1.80,
        "Performance": 1.40,
        "Billing": 0.80,
        "Feature Request": 72 / 24,
        "Account Management": 0.90,
        "Training / Onboarding": 1.20,
    }

    base_hours = (
        priority_base[priority]
        * issue_multiplier[issue_category]
    )

    resolution_hours = rng.lognormal(
        mean=np.log(base_hours),
        sigma=0.45,
    )

    return round(
        max(resolution_hours, 0.5),
        2,
    )


# =========================================================
# Main support generator
# =========================================================

def generate_support(
    customers: pd.DataFrame,
    customer_state: pd.DataFrame,
    subscriptions: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # Customer information
    # -----------------------------------------------------

    customer_info = customers[
        [
            "customer_id",
            "segment",
        ]
    ].drop_duplicates()

    # -----------------------------------------------------
    # Active contract windows
    # -----------------------------------------------------

    contract_dates = (
        subscriptions[
            [
                "customer_id",
                "start_date",
                "end_date",
            ]
        ]
        .assign(
            start_date=lambda df: pd.to_datetime(
                df["start_date"]
            ),
            end_date=lambda df: pd.to_datetime(
                df["end_date"]
            ),
        )
        .groupby("customer_id")
        .agg(
            customer_start=("start_date", "min"),
            customer_end=("end_date", "max"),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # Merge customer state with customer attributes
    # -----------------------------------------------------

    state = customer_state[
        [
            "customer_id",
            "week_start",
            "health_score",
            "behaviour",
        ]
    ].merge(
        customer_info,
        on="customer_id",
        how="left",
    )

    state = state.merge(
        contract_dates,
        on="customer_id",
        how="left",
    )

    state["week_start"] = pd.to_datetime(
        state["week_start"]
    )

    # -----------------------------------------------------
    # Only generate support during active contract periods.
    # -----------------------------------------------------

    state["is_active"] = (
        (state["week_start"] >= state["customer_start"])
        & (
            state["week_start"]
            <= state["customer_end"]
        )
    )

    state = state[
        state["is_active"]
    ].copy()

    # -----------------------------------------------------
    # Generate tickets week-by-week
    # -----------------------------------------------------

    support_rows = []

    ticket_counter = 1

    for row in state.itertuples(index=False):

        # -------------------------------------------------
        # Health-driven ticket intensity
        # -------------------------------------------------

        # Health 75 ≈ normal ticket volume.
        #
        # Lower health → more support demand.
        # Higher health → fewer support interactions.

        health_factor = np.clip(
            (100 - row.health_score) / 25,
            0.25,
            3.0,
        )

        # Convert monthly ticket rate to weekly rate.

        weekly_ticket_rate = (
            BASE_TICKETS_PER_MONTH[row.segment]
            / 4.345
            * (
                0.45
                + 0.55 * health_factor
            )
        )

        # Add customer/week randomness.

        weekly_ticket_rate *= rng.lognormal(
            mean=0,
            sigma=0.20,
        )

        ticket_count = rng.poisson(
            weekly_ticket_rate
        )

        if ticket_count == 0:
            continue

        # -------------------------------------------------
        # Issue probabilities
        # -------------------------------------------------

        issue_probabilities = _get_issue_probabilities(
            row.health_score
        )

        issue_values = list(
            issue_probabilities.keys()
        )

        issue_probs = list(
            issue_probabilities.values()
        )

        # -------------------------------------------------
        # Priority probabilities
        # -------------------------------------------------

        priority_probabilities = (
            _get_priority_probabilities(
                row.segment,
                row.health_score,
            )
        )

        priority_values = [
            "Low",
            "Medium",
            "High",
            "Critical",
        ]

        # -------------------------------------------------
        # Channel probabilities
        # -------------------------------------------------

        channel_values = list(
            CHANNEL_PROBABILITY.keys()
        )

        channel_probs = list(
            CHANNEL_PROBABILITY.values()
        )

        # -------------------------------------------------
        # Generate each ticket
        # -------------------------------------------------

        for _ in range(ticket_count):

            # ---------------------------------------------
            # Random timestamp during the week
            # ---------------------------------------------

            random_day = rng.integers(
                0,
                7,
            )

            random_hour = rng.integers(
                8,
                19,
            )

            random_minute = rng.integers(
                0,
                60,
            )

            created_at = (
                pd.Timestamp(row.week_start)
                + pd.Timedelta(
                    days=int(random_day)
                )
                + pd.Timedelta(
                    hours=int(random_hour)
                )
                + pd.Timedelta(
                    minutes=int(random_minute)
                )
            )

            # ---------------------------------------------
            # Issue category
            # ---------------------------------------------

            issue_category = _choose_weighted(
                rng,
                issue_values,
                issue_probs,
            )

            # ---------------------------------------------
            # Priority
            # ---------------------------------------------

            # -------------------------------------------------
# Priority
# -------------------------------------------------

            priority_values = list(
                PRIORITY_WEIGHTS[row.segment].keys()
            )

            priority_probs = list(
                PRIORITY_WEIGHTS[row.segment].values()
            )

            priority = _choose_weighted(
                rng,
                priority_values,
                priority_probs,
            )

            # ---------------------------------------------
            # Channel
            # ---------------------------------------------

            channel = _choose_weighted(
                rng,
                channel_values,
                channel_probs,
            )

            # ---------------------------------------------
            # Resolution time
            # ---------------------------------------------

            resolution_hours = (
                _generate_resolution_hours(
                    rng,
                    priority,
                    issue_category,
                )
            )

            resolved_at = (
                created_at
                + pd.Timedelta(
                    hours=resolution_hours
                )
            )

            # ---------------------------------------------
            # Escalation
            # ---------------------------------------------

            escalation_probability = {
                "Low": 0.03,
                "Medium": 0.08,
                "High": 0.20,
                "Critical": 0.45,
            }[priority]

            # Slightly higher escalation for unhealthy
            # customers.

            escalation_probability *= (
                0.75
                + 0.50
                * (
                    (100 - row.health_score)
                    / 100
                )
            )

            escalated = (
                rng.random()
                < np.clip(
                    escalation_probability,
                    0,
                    0.90,
                )
            )

            # ---------------------------------------------
            # Reopen probability
            # ---------------------------------------------

            reopen_probability = {
                "Low": 0.04,
                "Medium": 0.07,
                "High": 0.12,
                "Critical": 0.18,
            }[priority]

            reopened = (
                rng.random()
                < reopen_probability
            )

            # ---------------------------------------------
            # CSAT
            # ---------------------------------------------

            # Start around 4.3 / 5.
            #
            # Satisfaction decreases with:
            # - escalation
            # - reopening
            # - slow resolution
            # - lower customer health

            csat_base = 4.3

            if escalated:
                csat_base -= 0.5

            if reopened:
                csat_base -= 0.4

            if resolution_hours > 48:
                csat_base -= 0.3

            # Subtle health effect.

            csat_base += (
                (row.health_score - 75)
                / 100
            )

            csat_score = int(
                np.clip(
                    round(
                        rng.normal(
                            csat_base,
                            0.55,
                        )
                    ),
                    1,
                    5,
                )
            )

            # ---------------------------------------------
            # Final ticket record
            # ---------------------------------------------

            support_rows.append(
                {
                    "ticket_id": (
                        f"TKT{ticket_counter:08d}"
                    ),
                    "customer_id": row.customer_id,
                    "created_at": created_at,
                    "resolved_at": resolved_at,
                    "issue_category": issue_category,
                    "priority": priority,
                    "channel": channel,
                    "resolution_hours": resolution_hours,
                    "reopened": bool(reopened),
                    "csat_score": csat_score,
                    "escalated": bool(escalated),
                }
            )

            ticket_counter += 1

    # -----------------------------------------------------
    # Convert to DataFrame
    # -----------------------------------------------------

    support = pd.DataFrame(
        support_rows
    )

    # -----------------------------------------------------
    # Handle case where no tickets are generated
    # -----------------------------------------------------

    if support.empty:

        return pd.DataFrame(
            columns=[
                "ticket_id",
                "customer_id",
                "created_at",
                "resolved_at",
                "issue_category",
                "priority",
                "channel",
                "resolution_hours",
                "reopened",
                "csat_score",
                "escalated",
            ]
        )

    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    support = support.sort_values(
        [
            "customer_id",
            "created_at",
        ]
    ).reset_index(drop=True)

    return support