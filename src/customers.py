from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    INDUSTRIES,
    N_CUSTOMERS,
    RANDOM_SEED,
    REGIONS,
    RAW_DATA_DIR,
    SEGMENT_DISTRIBUTION,
    SIMULATION_START,
    SIMULATION_END,
)


def generate_customers(
    n_customers: int = N_CUSTOMERS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    customer_ids = [
        f"C{i:05d}"
        for i in range(1, n_customers + 1)
    ]

    # -------------------------
    # Customer segment
    # -------------------------

    segments = rng.choice(
        list(SEGMENT_DISTRIBUTION.keys()),
        size=n_customers,
        p=list(SEGMENT_DISTRIBUTION.values()),
    )

    # -------------------------
    # Company size
    # -------------------------

    company_size = np.zeros(n_customers, dtype=int)

    smb_mask = segments == "SMB"
    mid_mask = segments == "Mid-Market"
    ent_mask = segments == "Enterprise"

    company_size[smb_mask] = rng.integers(
        10,
        251,
        size=smb_mask.sum(),
    )

    company_size[mid_mask] = rng.integers(
        251,
        2001,
        size=mid_mask.sum(),
    )

    # Enterprise company sizes should be more right-skewed
    company_size[ent_mask] = np.clip(
        rng.lognormal(
            mean=np.log(5000),
            sigma=0.8,
            size=ent_mask.sum(),
        ).astype(int),
        2001,
        50_000,
    )

    # -------------------------
    # Industry
    # -------------------------

    industries = rng.choice(
        INDUSTRIES,
        size=n_customers,
    )

    # -------------------------
    # Region
    # -------------------------

    regions = rng.choice(
        REGIONS,
        size=n_customers,
        p=[0.35, 0.25, 0.20, 0.10, 0.10],
    )

    # -------------------------
    # Signup date
    # -------------------------

    start_date = pd.Timestamp(SIMULATION_START)
    end_date = pd.Timestamp(SIMULATION_END)

    days_range = (end_date - start_date).days

    signup_offsets = rng.integers(
        0,
        days_range + 1,
        size=n_customers,
    )

    signup_dates = (
        start_date
        + pd.to_timedelta(signup_offsets, unit="D")
    )

    # -------------------------
    # Acquisition channel
    # -------------------------

    acquisition_channels = rng.choice(
        [
            "Direct Sales",
            "Partner",
            "Inbound",
            "Enterprise Sales",
            "Referral",
        ],
        size=n_customers,
        p=[0.25, 0.15, 0.25, 0.20, 0.15],
    )

    # -------------------------
    # Account manager
    # -------------------------

    account_manager_ids = rng.choice(
        [f"AM_{i:03d}" for i in range(1, 101)],
        size=n_customers,
    )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "industry": industries,
            "region": regions,
            "segment": segments,
            "company_size": company_size,
            "signup_date": signup_dates,
            "account_manager_id": account_manager_ids,
            "acquisition_channel": acquisition_channels,
        }
    )

    return customers


if __name__ == "__main__":

    customers = generate_customers()

    output_path = RAW_DATA_DIR / "customers.csv"

    customers.to_csv(
        output_path,
        index=False,
    )

    print(f"Generated {len(customers):,} customers.")
    print(f"Saved to: {output_path}")

    print("\nSegment distribution:")
    print(
        customers["segment"]
        .value_counts(normalize=True)
        .round(3)
    )