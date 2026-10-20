"""Step 1 - Load and preprocess the house prices dataset."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/house_prices.csv"
OUT_DIR = "data/processed"
SCALER_PATH = "artifacts/scaler.joblib"

# The file has no header row, so we name the columns ourselves.
COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
           "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
TARGET = "MEDV"          # median house value in $1000s
DROP_COLS = ["B"]        # derived from the racial makeup of a neighbourhood
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path=RAW_PATH):
    """Read the space-separated file (no header) and name the columns."""
    return pd.read_csv(path, header=None, sep=r"\s+", names=COLUMNS)


def split_and_scale(df):
    """Random 80/20 split, then standardize features using the training set only."""
    df = df.drop(columns=DROP_COLS)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler


def main():
    df = load_data()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum())} | Duplicate rows: {int(df.duplicated().sum())}")
    capped = int((df[TARGET] == df[TARGET].max()).sum())
    print(f"Houses at the price cap (${df[TARGET].max():.0f}k): {capped}\n")

    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = split_and_scale(df)
    print(f"Features used: {X_train.shape[1]} (dropped: {DROP_COLS})")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    # Checks
    assert df.isna().sum().sum() == 0
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(X_test.index), "Train and test overlap"
    assert X_train_s.mean().abs().max() < 1e-9, "Scaled mean is not ~0"
    assert (X_train_s.std(ddof=0) - 1).abs().max() < 1e-9, "Scaled std is not ~1"
    print("Preprocessing verified")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(f"{OUT_DIR}/train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(f"{OUT_DIR}/test.csv", index=False)
    X_train_s.assign(**{TARGET: y_train}).to_csv(f"{OUT_DIR}/train_scaled.csv", index=False)
    X_test_s.assign(**{TARGET: y_test}).to_csv(f"{OUT_DIR}/test_scaled.csv", index=False)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved 4 CSVs to {OUT_DIR}/ and scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
