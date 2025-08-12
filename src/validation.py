# src/validation.py

import pandas as pd  # type: ignore

def validate_columns(df: pd.DataFrame, expected_columns: list) -> bool:
    """
    Validates if all expected columns exist in the dataset.
    """
    missing_cols = [col for col in expected_columns if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_columns]

    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    if extra_cols:
        print(f"⚠ Extra columns found (not in expected list): {extra_cols}")
    print("✅ All expected columns are present.")
    return True


def check_missing_values(df: pd.DataFrame) -> bool:
    """
    Checks for missing values in the dataset.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("❌ Missing values detected:")
        print(missing)
        return False
    print("✅ No missing values found.")
    return True
