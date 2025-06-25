from data_collection import load_customers, load_products, load_descriptions, merge_data
from cleaning import clean_data
from feature_engineering import feature_engineering

import pandas as pd
import os

def run_pipeline():
    # Step 1: Collect and integrate

    customers = load_customers()
    products = load_products()
    descriptions = load_descriptions()
    combined = merge_data(customers, products, descriptions)

    os.makedirs("data/raw", exist_ok=True)
    combined_path = "data/raw/combined_data.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[✓] Merged data saved to {combined_path}")

    # Step 2: Clean

    cleaned = clean_data(combined)
    cleaned_path = "data/processed/cleaned_data.csv"
    os.makedirs("data/processed", exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)
    print(f"[✓] Cleaned data saved to {cleaned_path}")

    # Step 3: Feature engineer

    transformed = feature_engineering(cleaned)
    transformed_path = "data/processed/transformed_features.csv"
    transformed.to_csv(transformed_path, index=False)
    print(f"[✓] Transformed features saved to {transformed_path}")

if __name__ == "__main__":
    run_pipeline()