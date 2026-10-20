"""Step 4 - Evaluate the linear regression model with R-squared and MSE."""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "artifacts/model_raw.joblib"
METRICS_PATH = "outputs/metrics.json"
CHART_PATH = "images/actual_vs_predicted.png"
TARGET = "MEDV"


def load_xy(path):
    df = pd.read_csv(path)
    return df.drop(columns=TARGET), df[TARGET]


def compute_metrics(y_true, y_pred):
    """R-squared, MSE, RMSE and MAE. Prices are in $1000s."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def plot_actual_vs_predicted(y_true, y_pred, r2):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, color="#2563eb", alpha=0.7, s=50)
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, color="#f97316", lw=2, ls="--", label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual price ($1000s)")
    ax.set_ylabel("Predicted price ($1000s)")
    ax.set_title(f"Predicted vs actual on unseen houses (R\u00b2 = {r2:.2f})",
                 fontsize=16, weight="bold", loc="left")
    ax.legend(fontsize=13, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    model = joblib.load(MODEL_PATH)
    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)

    train_m = compute_metrics(y_train, model.predict(X_train))
    test_pred = model.predict(X_test)
    test_m = compute_metrics(y_test, test_pred)

    # Baseline: always guess the average training price
    baseline_mse = mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))

    print(f"{'':<10}{'R-squared':>10}{'MSE':>10}{'RMSE':>10}{'MAE':>10}")
    for name, m in (("Train", train_m), ("Test", test_m)):
        print(f"{name:<10}{m['r2']:>10.3f}{m['mse']:>10.2f}{m['rmse']:>10.2f}{m['mae']:>10.2f}")
    print(f"\nBaseline (always guess the average): test MSE = {baseline_mse:.2f}")
    print(f"Model cuts the error by {1 - test_m['mse'] / baseline_mse:.0%}")
    print(f"Typical miss: about ${test_m['rmse'] * 1000:,.0f} per house (RMSE)")

    # Checks
    assert len(test_pred) == len(y_test)
    assert test_m["mse"] < baseline_mse, "Model should beat the baseline"
    assert abs(test_m["rmse"] ** 2 - test_m["mse"]) < 1e-9
    print("\nEvaluation verified")

    os.makedirs("outputs", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"train": train_m, "test": test_m, "baseline_test_mse": baseline_mse}, f, indent=2)
    plot_actual_vs_predicted(y_test, test_pred, test_m["r2"])
    print(f"Saved -> {METRICS_PATH} and {CHART_PATH}")


if __name__ == "__main__":
    main()
