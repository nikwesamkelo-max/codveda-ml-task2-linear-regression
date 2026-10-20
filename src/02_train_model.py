"""Step 2 - Train a linear regression model on the house prices data."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

TRAIN_PATH = "data/processed/train.csv"
TRAIN_SCALED_PATH = "data/processed/train_scaled.csv"
MODEL_PATH = "artifacts/model_raw.joblib"
MODEL_SCALED_PATH = "artifacts/model_scaled.joblib"
TARGET = "MEDV"


def load_xy(path):
    """Split a processed CSV into features (X) and target (y)."""
    df = pd.read_csv(path)
    return df.drop(columns=TARGET), df[TARGET]


def train_linear_regression(X, y):
    """Fit ordinary least squares: finds the intercept and one coefficient per
    feature that minimize the squared prediction errors on the training data."""
    return LinearRegression().fit(X, y)


def main():
    X_train, y_train = load_xy(TRAIN_PATH)
    X_train_s, _ = load_xy(TRAIN_SCALED_PATH)
    print(f"Training on {len(X_train)} houses with {X_train.shape[1]} features\n")

    # Model A: raw units (coefficients are easy to explain, e.g. "per extra room")
    model = train_linear_regression(X_train, y_train)
    # Model B: standardized features (coefficients are comparable across features)
    model_scaled = train_linear_regression(X_train_s, y_train)

    print(f"Intercept (raw model): {model.intercept_:.3f}")
    print(f"Training R-squared: {model.score(X_train, y_train):.3f}\n")

    # Checks
    assert model.n_features_in_ == X_train.shape[1]
    assert not np.isnan(model.coef_).any()
    assert np.allclose(model.predict(X_train), model_scaled.predict(X_train_s)), \
        "Raw and scaled models should predict identically"
    print("Both models give identical predictions (scaling only changes the coefficients)")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model_scaled, MODEL_SCALED_PATH)
    print(f"Saved -> {MODEL_PATH} and {MODEL_SCALED_PATH}")
    print("Test data untouched: it is used only in step 4")


if __name__ == "__main__":
    main()
