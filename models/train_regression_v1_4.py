import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT NINJA - V1.4
# NEXT-DAY RETURN REGRESSION
# ============================================================

print("=" * 65)
print("PROJECT NINJA - V1.4")
print("RANDOM FOREST RETURN REGRESSION")
print("=" * 65)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    "data/train_v1.csv",
    index_col=0,
    parse_dates=True
)

validation = pd.read_csv(
    "data/validation_v1.csv",
    index_col=0,
    parse_dates=True
)

test = pd.read_csv(
    "data/test_v1.csv",
    index_col=0,
    parse_dates=True
)


print("\nTraining:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Return_1D",
    "Return_5D",
    "Return_20D",
    "Distance_MA20",
    "Distance_MA50",
    "Distance_MA200",
    "RSI",
    "Volatility_20D",
    "Volume_Ratio"
]


# ============================================================
# TARGET
# ============================================================

# IMPORTANT:
#
# train_v1.csv does not currently contain Future_Return.
#
# We will reconstruct it from actual market prices.


import yfinance as yf


print("\nDownloading market data...")

market = yf.Ticker("RELIANCE.NS").history(
    period="15y"
)


market.index = pd.to_datetime(
    market.index
)

if market.index.tz is not None:
    market.index = market.index.tz_localize(None)


# ============================================================
# CREATE FUTURE RETURN
# ============================================================

market["Future_Return"] = (
    market["Close"].shift(-1)
    / market["Close"]
    - 1
)


# ============================================================
# ALIGN DATASETS
# ============================================================

def prepare_dataset(dataset):

    dataset = dataset.copy()

    dataset.index = pd.to_datetime(
        dataset.index
    )

    if dataset.index.tz is not None:
        dataset.index = dataset.index.tz_localize(None)

    dataset = dataset.join(
        market[["Future_Return"]],
        how="left"
    )

    dataset = dataset.dropna(
        subset=FEATURES + ["Future_Return"]
    )

    return dataset


train = prepare_dataset(train)
validation = prepare_dataset(validation)
test = prepare_dataset(test)


print("\nPrepared datasets:")

print("Training:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# ============================================================
# INPUT / TARGET
# ============================================================

X_train = train[FEATURES]
y_train = train["Future_Return"]

X_validation = validation[FEATURES]
y_validation = validation["Future_Return"]

X_test = test[FEATURES]
y_test = test["Future_Return"]


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 65)
print("BASELINE")
print("=" * 65)


# Predict the historical training-set average
baseline_prediction = y_train.mean()


baseline_validation_predictions = np.full(
    len(y_validation),
    baseline_prediction
)

baseline_test_predictions = np.full(
    len(y_test),
    baseline_prediction
)


baseline_mae = mean_absolute_error(
    y_test,
    baseline_test_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_test_predictions
    )
)


print(
    f"\nTraining average return: "
    f"{baseline_prediction * 100:.4f}%"
)

print(
    f"Baseline test MAE: "
    f"{baseline_mae * 100:.4f}%"
)

print(
    f"Baseline test RMSE: "
    f"{baseline_rmse * 100:.4f}%"
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 65)
print("TRAINING RANDOM FOREST")
print("=" * 65)


model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Training complete.")


# ============================================================
# VALIDATION
# ============================================================

validation_predictions = model.predict(
    X_validation
)


validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)


print("\n" + "=" * 65)
print("VALIDATION RESULTS")
print("=" * 65)

print(
    f"\nMAE: "
    f"{validation_mae * 100:.4f}%"
)

print(
    f"RMSE: "
    f"{validation_rmse * 100:.4f}%"
)

print(
    f"R²: "
    f"{validation_r2:.4f}"
)


# ============================================================
# TEST
# ============================================================

test_predictions = model.predict(
    X_test
)


test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)


print("\n" + "=" * 65)
print("TEST RESULTS")
print("=" * 65)

print(
    f"\nMAE: "
    f"{test_mae * 100:.4f}%"
)

print(
    f"RMSE: "
    f"{test_rmse * 100:.4f}%"
)

print(
    f"R²: "
    f"{test_r2:.4f}"
)


# ============================================================
# PREDICTIONS
# ============================================================

results = test.copy()

results["Predicted_Return"] = test_predictions


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("PREDICTION ANALYSIS")
print("=" * 65)


print(
    f"\nMinimum predicted return: "
    f"{test_predictions.min() * 100:.4f}%"
)

print(
    f"Maximum predicted return: "
    f"{test_predictions.max() * 100:.4f}%"
)

print(
    f"Average predicted return: "
    f"{test_predictions.mean() * 100:.4f}%"
)


# Correlation between prediction and actual return

correlation = np.corrcoef(
    test_predictions,
    y_test
)[0, 1]


print(
    f"\nPrediction / actual correlation: "
    f"{correlation:.4f}"
)


# ============================================================
# TOP PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("TOP 20 PREDICTED RETURNS")
print("=" * 65)


top_predictions = results.sort_values(
    "Predicted_Return",
    ascending=False
).head(20)


print(
    top_predictions[
        [
            "Predicted_Return",
            "Future_Return"
        ]
    ].to_string()
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    "data/test_regression_predictions_v1_4.csv"
)


print("\nSaved:")
print(
    "data/test_regression_predictions_v1_4.csv"
)


print("\n" + "=" * 65)
print("V1.4 COMPLETE")
print("=" * 65)