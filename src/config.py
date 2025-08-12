# src/config.py

# Path to raw dataset
RAW_DATA_PATH = "data/raw/winequality-red.csv"

# URL to download dataset if not found
RAW_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Expected columns for validation
EXPECTED_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]
