import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from datetime import datetime
import os
from pathlib import Path

# Get project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
TRANSFORMED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transformed_features.csv"


def feature_engineering(df):
    # 1. Create new features

    df["total_spend"] = df["price"] * df["stock_level"]
    df["days_since_signup"] = (
        pd.to_datetime("today") - pd.to_datetime(df["signup_date"])
    ).dt.days

    # 2. Encode country (Label encoding for now)

    label_encoder = LabelEncoder()
    df["country_encoded"] = label_encoder.fit_transform(df["country"])

    # 3. One-hot encode product category

    df = pd.get_dummies(df, columns=["category"], prefix="cat")

    # 4. Scale numeric features

    scaler = StandardScaler()
    scaled_cols = ["price", "stock_level", "total_spend", "days_since_signup"]
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

    return df


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(CLEANED_DATA_PATH)
    transformed_df = feature_engineering(df)
    transformed_df.to_csv(TRANSFORMED_DATA_PATH, index=False)
    print(f"[✓] Transformed features saved to {TRANSFORMED_DATA_PATH}")
