import numpy as np
import pandas as pd

from src.config import (
    CONTRACT_TYPES,
    RANDOM_SEED,
    RAW_DATA_DIR,
)


# ---------------------------------------------------------
# Contract configuration
# ---------------------------------------------------------

CONTRACT_TYPE_BY_SEGMENT = {
    "SMB": {
        "Annual": 0.75,
        "Two-year": 0.20,
        "Three-year": 0.05,
    },
    "Mid-Market": {
        "Annual": 0.65,
        "Two-year": 0.25,
        "Three-year": 0.10,
    },
    "Enterprise": {
        "Annual": 0.50,
        "Two-year": 0.30,
        "Three-year": 0.20,
    },
}


AUTO_RENEW_PROBABILITY = {
    "SMB": 0.70,
    "Mid-Market": 0.78,
    "Enterprise": 0.85,
}


# ACV parameters:
# median and sigma for the log-normal distribution.
ACV_PARAMETERS = {
    "SMB": {
        "median": 500_000,
        "sigma": 0.65,
        "min": 200_000,
        "max": 1_500_000,
    },
    "Mid-Market": {
        "median": 3_000_000,
        "sigma": 0.70,
        "min": 1_000_000,
        "max": 10_000_000,
    },
    "Enterprise": {
        "median": 15_000_000,
        "sigma": 0.65,
        "min": 7_500_000,
        "max": 100_000_000,
    },
}


def generate_contracts(
    customers: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    contracts = customers[
        [
            "customer_id",
            "segment",
            "signup_date",
        ]
    ].copy()

    # -----------------------------------------------------
    # Contract ID
    # -----------------------------------------------------

    contracts["contract_id"] = [
        f"CT{i:06d}"
        for i in range(1, len(contracts) + 1)
    ]

    # -----------------------------------------------------
    # Contract type
    # -----------------------------------------------------

    # -----------------------------------------------------
# Contract type
# -----------------------------------------------------

    contracts["contract_type"] = None

    for segment, distribution in CONTRACT_TYPE_BY_SEGMENT.items():

        mask = contracts["segment"] == segment

        contracts.loc[mask, "contract_type"] = rng.choice(
            list(distribution.keys()),
            size=mask.sum(),
            p=list(distribution.values()),
        )

    # -----------------------------------------------------
    # Annual Contract Value
    # -----------------------------------------------------

    acv_values = []

    for segment in contracts["segment"]:

        median = ACV_PARAMETERS[segment]["median"]
        sigma = ACV_PARAMETERS[segment]["sigma"]

        # For a log-normal distribution:
        #
        # median = exp(mu)
        #
        # therefore:
        # mu = log(median)

        mu = np.log(median)

        acv = rng.lognormal(
            mean=mu,
            sigma=sigma,
        )

        acv = np.clip(
            acv,
            ACV_PARAMETERS[segment]["min"],
            ACV_PARAMETERS[segment]["max"],
        )

        acv_values.append(acv)

    contracts["annual_contract_value"] = np.round(
        acv_values,
        2,
    )

    # -----------------------------------------------------
    # Contract dates
    # -----------------------------------------------------

    contracts["start_date"] = pd.to_datetime(
        contracts["signup_date"]
    )

    duration_years = contracts["contract_type"].map(
        {
            "Annual": 1,
            "Two-year": 2,
            "Three-year": 3,
        }
    )

    contracts["end_date"] = (
        contracts["start_date"]
        + pd.to_timedelta(
            duration_years * 365,
            unit="D",
        )
    )

    contracts["renewal_date"] = contracts["end_date"]

    # -----------------------------------------------------
    # Auto-renew
    # -----------------------------------------------------

    contracts["auto_renew"] = [
        rng.random()
        < AUTO_RENEW_PROBABILITY[segment]
        for segment in contracts["segment"]
    ]

    # -----------------------------------------------------
    # Contract status
    # -----------------------------------------------------

    contracts["contract_status"] = "Active"

    # -----------------------------------------------------
    # Total Contract Value
    # -----------------------------------------------------

    contracts["total_contract_value"] = np.round(
        contracts["annual_contract_value"]
        * duration_years,
        2,
    )

    # -----------------------------------------------------
    # Final column ordering
    # -----------------------------------------------------

    contracts = contracts[
        [
            "contract_id",
            "customer_id",
            "segment",
            "start_date",
            "end_date",
            "renewal_date",
            "annual_contract_value",
            "total_contract_value",
            "contract_type",
            "auto_renew",
            "contract_status",
        ]
    ]

    return contracts


if __name__ == "__main__":

    customers = pd.read_csv(
        RAW_DATA_DIR / "customers.csv",
        parse_dates=["signup_date"],
    )

    contracts = generate_contracts(customers)

    output_path = RAW_DATA_DIR / "contracts.csv"

    contracts.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Generated {len(contracts):,} contracts."
    )

    print(
        f"Saved to: {output_path}"
    )