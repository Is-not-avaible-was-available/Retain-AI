import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_DATA_DIR


PRODUCT_ADOPTION_PROBABILITY = {
    "SMB": {
        "P002": 0.35,
        "P003": 0.25,
        "P004": 0.20,
        "P005": 0.15,
    },
    "Mid-Market": {
        "P002": 0.55,
        "P003": 0.50,
        "P004": 0.40,
        "P005": 0.35,
    },
    "Enterprise": {
        "P002": 0.75,
        "P003": 0.70,
        "P004": 0.65,
        "P005": 0.55,
    },
}


def generate_subscriptions(
    customers: pd.DataFrame,
    contracts: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # Merge the information needed to generate subscriptions.
    customer_contracts = customers[
        [
            "customer_id",
            "segment",
        ]
    ].merge(
        contracts[
            [
                "customer_id",
                "start_date",
                "end_date",
                "annual_contract_value",
            ]
        ],
        on="customer_id",
        how="inner",
    )

    subscription_rows = []

    subscription_counter = 1

    optional_products = [
        "P002",
        "P003",
        "P004",
        "P005",
    ]

    for _, customer in customer_contracts.iterrows():

        customer_id = customer["customer_id"]
        segment = customer["segment"]
        customer_acv = customer["annual_contract_value"]

        # ---------------------------------------------
        # Core Platform is mandatory
        # ---------------------------------------------

        selected_products = ["P001"]

        # ---------------------------------------------
        # Select optional products
        # ---------------------------------------------

        for product_id in optional_products:

            probability = PRODUCT_ADOPTION_PROBABILITY[
                segment
            ][product_id]

            if rng.random() < probability:
                selected_products.append(product_id)

        # ---------------------------------------------
        # Generate product allocation weights
        # ---------------------------------------------

        weights = rng.dirichlet(
            np.ones(len(selected_products)) * 3
        )

        # ---------------------------------------------
        # Allocate ACV across products
        # ---------------------------------------------

        allocated_values = [
            round(customer_acv * weight, 2)
            for weight in weights
        ]

        # ---------------------------------------------
        # Correct rounding residual
        #
        # Ensures:
        #
        # sum(product annual values)
        # ==
        # customer annual contract value
        # ---------------------------------------------

        rounding_difference = round(
            customer_acv - sum(allocated_values),
            2,
        )

        allocated_values[-1] = round(
            allocated_values[-1] + rounding_difference,
            2,
        )

        # ---------------------------------------------
        # Create subscriptions
        # ---------------------------------------------

        for product_id, annual_value in zip(
            selected_products,
            allocated_values,
        ):

            monthly_value = round(
                annual_value / 12,
                2,
            )

            subscription_rows.append(
                {
                    "subscription_id": (
                        f"SUB{subscription_counter:07d}"
                    ),
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "start_date": customer["start_date"],
                    "end_date": customer["end_date"],
                    "annual_value": annual_value,
                    "monthly_value": monthly_value,
                    "status": "Active",
                }
            )

            subscription_counter += 1

    subscriptions = pd.DataFrame(
        subscription_rows
    )

    return subscriptions


if __name__ == "__main__":

    customers = pd.read_csv(
        RAW_DATA_DIR / "customers.csv"
    )

    contracts = pd.read_csv(
        RAW_DATA_DIR / "contracts.csv"
    )

    subscriptions = generate_subscriptions(
        customers,
        contracts,
    )

    output_path = (
        RAW_DATA_DIR / "subscriptions.csv"
    )

    subscriptions.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Generated {len(subscriptions):,} subscriptions."
    )

    print(
        f"Saved to: {output_path}"
    )