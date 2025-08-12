# src/processing.py

import pandas as pd
import os

PROCESSED_DATA_PATH = "data/processed/winequality-red.csv"

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on the dataset and return processed DataFrame.
    """
    # Example cleaning — remove duplicates
    df = df.drop_duplicates()

    # If needed: handle missing values (we already validated so unlikely)
    df = df.fillna(0)

    return df

def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    """
    Save processed DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to {path}")
