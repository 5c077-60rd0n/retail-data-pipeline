from data_collection import load_customers, load_products, load_descriptions, merge_data
from cleaning import clean_data
from feature_engineering import feature_engineering

import pandas as pd
import os
from pathlib import Path

# Get project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent

def run_pipeline():
    # Step 1: Collect and integrate

    customers = load_customers()
    products = load_products()
    descriptions = load_descriptions()
    combined = merge_data(customers, products, descriptions)

    # Ensure directories exist
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    combined_path = raw_dir / "combined_data.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[✓] Merged data saved to {combined_path}")

    # Step 2: Clean

    cleaned = clean_data(combined)
    
    # Ensure processed directory exists
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_path = processed_dir / "cleaned_data.csv"
    cleaned.to_csv(cleaned_path, index=False)
    print(f"[✓] Cleaned data saved to {cleaned_path}")

    # Step 3: Feature engineer

    transformed = feature_engineering(cleaned)
    transformed_path = processed_dir / "transformed_features.csv"
    transformed.to_csv(transformed_path, index=False)
    print(f"[✓] Transformed features saved to {transformed_path}")

if __name__ == "__main__":
    run_pipeline()