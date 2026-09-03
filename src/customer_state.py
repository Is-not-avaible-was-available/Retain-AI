import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

HEALTH_MIN = 0
HEALTH_MAX = 100

BASELINE_HEALTH_MEAN = 75
BASELINE_HEALTH_STD = 10

HEALTH_BEHAVIOURS = {
    "Improving": {
        "probability": 0.15,
        "trend_mean": 0.30,
        "trend_std": 0.06,
    },
    "Stable": {
        "probability": 0.60,
        "trend_mean": 0.00,
        "trend_std": 0.04,
    },
    "Declining": {
        "probability": 0.20,
        "trend_mean": -0.25,
        "trend_std": 0.06,
    },
    "Rapidly Declining": {
        "probability": 0.05,
        "trend_mean": -0.50,
        "trend_std": 0.08,
    },
}


def generate_customer_state(
    customers: pd.DataFrame,
    start_date: str,
    end_date: str,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # Generate weekly dates
    # -----------------------------------------------------

    weeks = pd.date_range(
        start=start_date,
        end=end_date,
        freq="W-MON",
    )

    # -----------------------------------------------------
    # Customer-level latent attributes
    # -----------------------------------------------------

    customer_attributes = customers[
        [
            "customer_id",
        ]
    ].copy()

    n_customers = len(customer_attributes)

    # Baseline health
    baseline_health = np.clip(
        rng.normal(
            BASELINE_HEALTH_MEAN,
            BASELINE_HEALTH_STD,
            size=n_customers,
        ),
        40,
        100,
    )

    customer_attributes["baseline_health"] = (
        baseline_health
    )

    # Behaviour type
    behaviour_types = list(
        HEALTH_BEHAVIOURS.keys()
    )

    behaviour_probabilities = [
        HEALTH_BEHAVIOURS[b]["probability"]
        for b in behaviour_types
    ]

    customer_attributes["behaviour"] = rng.choice(
        behaviour_types,
        size=n_customers,
        p=behaviour_probabilities,
    )

    # Customer-specific trend
    trends = []

    for behaviour in customer_attributes[
        "behaviour"
    ]:

        parameters = HEALTH_BEHAVIOURS[
            behaviour
        ]

        trend = rng.normal(
            parameters["trend_mean"],
            parameters["trend_std"],
        )

        trends.append(trend)

    customer_attributes["health_trend"] = trends

    # Customer-specific volatility
    customer_attributes["volatility"] = rng.uniform(
        0.4,
        1.2,
        size=n_customers,
    )

    # -----------------------------------------------------
    # Generate weekly state
    # -----------------------------------------------------

    state_rows = []

    for _, customer in customer_attributes.iterrows():

        health = customer["baseline_health"]

        for week in weeks:

            # Weekly random shock
            noise = rng.normal(
                0,
                customer["volatility"],
            )

            health = (
                health
                + customer["health_trend"]
                + noise
            )

            # Keep health within bounds
            health = np.clip(
                health,
                HEALTH_MIN,
                HEALTH_MAX,
            )

            state_rows.append(
                {
                    "customer_id": customer[
                        "customer_id"
                    ],
                    "week_start": week,
                    "health_score": round(
                        health,
                        2,
                    ),
                    "baseline_health": round(
                        customer[
                            "baseline_health"
                        ],
                        2,
                    ),
                    "health_trend": round(
                        customer[
                            "health_trend"
                        ],
                        4,
                    ),
                    "volatility": round(
                        customer[
                            "volatility"
                        ],
                        4,
                    ),
                    "behaviour": customer[
                        "behaviour"
                    ],
                }
            )

    return pd.DataFrame(state_rows)