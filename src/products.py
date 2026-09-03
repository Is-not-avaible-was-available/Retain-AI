import pandas as pd

from src.config import PRODUCTS, RAW_DATA_DIR


PRODUCT_DESCRIPTIONS = {
    "P001": "Core Platform",
    "P002": "Analytics",
    "P003": "Workflow Automation",
    "P004": "AI Assistant",
    "P005": "Advanced Reporting",
}


PRODUCT_CATEGORIES = {
    "P001": "Core",
    "P002": "Analytics",
    "P003": "Automation",
    "P004": "AI",
    "P005": "Analytics",
}


BASE_MONTHLY_PRICES = {
    "P001": 25_000,
    "P002": 15_000,
    "P003": 20_000,
    "P004": 18_000,
    "P005": 12_000,
}


def generate_products() -> pd.DataFrame:

    products = pd.DataFrame(
        {
            "product_id": list(PRODUCTS.keys()),
            "product_name": [
                PRODUCT_DESCRIPTIONS[p]
                for p in PRODUCTS.keys()
            ],
            "category": [
                PRODUCT_CATEGORIES[p]
                for p in PRODUCTS.keys()
            ],
            "base_monthly_price": [
                BASE_MONTHLY_PRICES[p]
                for p in PRODUCTS.keys()
            ],
        }
    )

    return products


if __name__ == "__main__":

    products = generate_products()

    output_path = RAW_DATA_DIR / "products.csv"

    products.to_csv(
        output_path,
        index=False,
    )

    print(f"Generated {len(products)} products.")
    print(f"Saved to: {output_path}")