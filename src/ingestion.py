# src/ingestion.py

import os
import pandas as pd
import requests  # type: ignore
from src.config import RAW_DATA_PATH, RAW_DATA_URL, EXPECTED_COLUMNS
from src.validation import validate_columns, check_missing_values
from src.processing import process_data, save_processed_data

def download_dataset():
    """
    Downloads dataset from RAW_DATA_URL if RAW_DATA_PATH does not exist.
    """
    if os.path.exists(RAW_DATA_PATH):
        print("📂 Dataset already exists. Skipping download.")
        return
    print("⬇ Downloading dataset...")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    response = requests.get(RAW_DATA_URL)
    with open(RAW_DATA_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Download complete.")

def load_raw_data():
    """
    Loads the raw dataset into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(RAW_DATA_PATH, sep=";")
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

if __name__ == "__main__":
    # Step 1: Download if needed
    download_dataset()

    # Step 2: Load data
    df = load_raw_data()
    if df is None:
        print("❌ Data loading failed.")
        exit(1)

    # Step 3: Validate
    if not validate_columns(df, EXPECTED_COLUMNS):
        exit(1)
    if not check_missing_values(df):
        exit(1)

    # Step 4: Process
    df_processed = process_data(df)

    # Step 5: Save
    save_processed_data(df_processed)

    print("🎉 Data ingestion + validation + processing successful!")
