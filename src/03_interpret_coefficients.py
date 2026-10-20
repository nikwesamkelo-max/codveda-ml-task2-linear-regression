"""Step 3 - Interpret the linear regression coefficients."""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_PATH = "data/processed/train.csv"
MODEL_PATH = "artifacts/model_raw.joblib"
MODEL_SCALED_PATH = "artifacts/model_scaled.joblib"
SCALER_PATH = "artifacts/scaler.joblib"
TABLE_PATH = "outputs/coefficients.csv"
CHART_PATH = "images/coefficients.png"
TARGET = "MEDV"

DESCRIPTIONS = {
    "CRIM": "crime rate per capita", "ZN": "% land zoned for large lots",
    "INDUS": "% non-retail business acres", "CHAS": "borders Charles River (0/1)",
    "NOX": "nitric oxide concentration", "RM": "average rooms per home",
    "AGE": "% homes built before 1940", "DIS": "distance to employment centres",
    "RAD": "highway accessibility index", "PTRATIO": "pupil-teacher ratio",
    "TAX": "property tax rate", "LSTAT": "% lower-status population",
}


def build_coefficient_table(model, model_scaled, scaler, features):
    """One row per feature: raw coefficient, effect of a 1-std change, and std."""
    table = pd.DataFrame({
        "feature": features,
        "description": [DESCRIPTIONS[f] for f in features],
        "coef_raw": model.coef_,                  # change in MEDV ($1000s) per +1 unit
        "coef_per_1std": model_scaled.coef_,      # change in MEDV ($1000s) per +1 std
        "feature_std": scaler.scale_,
    })
    table["abs_effect"] = table["coef_per_1std"].abs()
    return table.sort_values("abs_effect", ascending=False).reset_index(drop=True)


def plain_english(row):
    direction = "raises" if row["coef_per_1std"] > 0 else "lowers"
    dollars = abs(row["coef_per_1std"]) * 1000
    return (f"{row['feature']:<8} +1 std (about {row['feature_std']:.2f}) "
            f"{direction} price by ${dollars:,.0f}  [{row['description']}]")


def plot_effects(table):
    t = table.sort_values("coef_per_1std")
    colors = ["#16a34a" if v > 0 else "#f97316" for v in t["coef_per_1std"]]
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.barh(t["feature"], t["coef_per_1std"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Change in house price ($1000s) for a 1-standard-deviation increase")
    ax.set_title("What drives house prices in this model?", fontsize=20, weight="bold", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    X = pd.read_csv(TRAIN_PATH).drop(columns=TARGET)
    model = joblib.load(MODEL_PATH)
    model_scaled = joblib.load(MODEL_SCALED_PATH)
    scaler = joblib.load(SCALER_PATH)

    table = build_coefficient_table(model, model_scaled, scaler, list(X.columns))

    # Check: a standardized coefficient is the raw one times the feature's std
    assert np.allclose(table["coef_per_1std"], table["coef_raw"] * table["feature_std"]), \
        "Raw and standardized coefficients do not match"
    print("Verified: standardized coefficient = raw coefficient x feature std\n")

    print("Note: the intercept alone is not meaningful (no house has 0 rooms).\n")
    print("Features ranked by impact (holding all others fixed):")
    for _, row in table.iterrows():
        print("  " + plain_english(row))

    os.makedirs("outputs", exist_ok=True)
    table.drop(columns="abs_effect").round(4).to_csv(TABLE_PATH, index=False)
    plot_effects(table)
    print(f"\nSaved -> {TABLE_PATH} and {CHART_PATH}")


if __name__ == "__main__":
    main()
