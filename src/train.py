# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

PROCESSED_DATA_PATH = "data/processed/winequality-red.csv"
MODEL_PATH = "models/model.joblib"

def train_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
