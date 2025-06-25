import pandas as pd
import numpy as np
from datetime import datetime

RAW_DATA_PATH = "data/raw/combined_data.csv"
CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"


def clean_data(df):
    # 1. Handle missing values

    df.fillna(
        {"description": "Nod description available", "price": df["price"].median()},
        inplace=True,
    )

    # 2. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 3. Standardize date formats
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    # 4. Currency normalization (e.g., round places to 2 decimals)

    df["price"] = df["price"].round(2)

    # 5. Detect outliers (e.g. prices beyond z-score of 3)

    price_zscores = (df["price"] - df["price"].mean()) / df["price"].std()
    df["outliers_price"] = price_zscores.abs() > 3

    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    cleaned_df = clean_data(df)

    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[✓] Cleaned dataset saved to {CLEANED_DATA_PATH}")
