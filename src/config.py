import os

# Paths for raw and processed data
RAW_DATA_PATH = "data/raw/winequality-red.csv"
PROCESSED_DATA_PATH = "data/processed/winequality-red.csv"

# (Optional) Dataset download URL if needed
RAW_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Columns expected in dataset
EXPECTED_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]
