import pandas as pd
import json
import os

# Paths

CUSTOMERS_CSV = "data/raw/customers.csv"
PRODUCTS_JSON = "data/raw/products.json"
DESCRIPTIONS_TXT = "data/raw/product_descriptions.txt"
OUTPUT_PATH = "data/raw/combined_data.csv"


def load_customers():
    return pd.read_csv(CUSTOMERS_CSV)


def load_products():
    with open(PRODUCTS_JSON, "r") as f:
        products = json.load(f)
    return pd.DataFrame(products)


def load_descriptions():
    descriptions = {}
    with open(DESCRIPTIONS_TXT, "r") as f:
        for line in f:
            try:
                pid, desc = line.strip().split("::", 1)
                descriptions[pid] = desc
            except ValueError:
                continue
    return (
        pd.DataFrame.from_dict(descriptions, orient="index", columns=["description"])
        .reset_index()
        .rename(columns={"index": "product_id"})
    )


def merge_data(customers, products, descriptions):
    merged = pd.merge(products, descriptions, on="product_id", how="left")
    # Simulate customer-product linkage (not realistic but illustrative)

    merged["customer_id"] = ["C001", "C002", "C003", "C004", "C005"]
    return pd.merge(merged, customers, on="customer_id", how="left")


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    customers_df = load_customers()
    products_df = load_products()
    descriptions_df = load_descriptions()

    combined_df = merge_data(customers_df, products_df, descriptions_df)
    combined_df.to_csv(OUTPUT_PATH, index=False)

    print(f"[✓] Combined dataset saved to {OUTPUT_PATH}")
