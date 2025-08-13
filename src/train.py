# src/train.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

from src.config import PROCESSED_DATA_PATH

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "wine_quality_model.pkl")


def load_processed_data():
    """Load processed dataset from CSV."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}")
    return pd.read_csv(PROCESSED_DATA_PATH)


def prepare_data(df):
    """
    Convert wine quality to binary labels.
    Good (1) if quality >= 7, else Bad (0).
    """
    df["quality_label"] = (df["quality"] >= 7).astype(int)
    X = df.drop(columns=["quality", "quality_label"])
    y = df["quality_label"]
    return X, y


def train_model(X_train, y_train):
    """Train XGBoost classifier."""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def save_model(model):
    """Save trained model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    print("📥 Loading processed data...")
    df = load_processed_data()

    print("🔄 Preparing data for binary classification...")
    X, y = prepare_data(df)

    print("✂ Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 Training model...")
    model = train_model(X_train, y_train)

    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("💾 Saving model...")
    save_model(model)

    print("🎉 Training complete!")
