import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================
# PROJECT NINJA - V1.0 ML MODEL
# ============================================

print("=" * 60)
print("PROJECT NINJA - V1.0 ML MODEL")
print("=" * 60)


# ============================================
# LOAD DATA
# ============================================

print("\nLoading datasets...")

train = pd.read_csv(
    "data/train_v1.csv",
    index_col=0
)

validation = pd.read_csv(
    "data/validation_v1.csv",
    index_col=0
)

test = pd.read_csv(
    "data/test_v1.csv",
    index_col=0
)


print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(validation)}")
print(f"Testing samples: {len(test)}")


# ============================================
# FEATURES
# ============================================

features = [
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


X_train = train[features]
y_train = train["Target"]

X_validation = validation[features]
y_validation = validation["Target"]

X_test = test[features]
y_test = test["Target"]


# ============================================
# BASELINE
# ============================================

print("\n" + "=" * 60)
print("BASELINES")
print("=" * 60)


# Always predict 0
always_down = [0] * len(y_test)

baseline_down_accuracy = accuracy_score(
    y_test,
    always_down
)

print(
    f"\nAlways predict 0 accuracy: "
    f"{baseline_down_accuracy * 100:.2f}%"
)


# Always predict 1
always_up = [1] * len(y_test)

baseline_up_accuracy = accuracy_score(
    y_test,
    always_up
)

print(
    f"Always predict 1 accuracy: "
    f"{baseline_up_accuracy * 100:.2f}%"
)


# ============================================
# MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)


model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000
        )
    )
])


model.fit(
    X_train,
    y_train
)


print("Model training complete.")


# ============================================
# VALIDATION
# ============================================

validation_predictions = model.predict(
    X_validation
)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(
    f"\nValidation accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


print("\nValidation classification report:")

print(
    classification_report(
        y_validation,
        validation_predictions
    )
)


# ============================================
# TEST
# ============================================

test_predictions = model.predict(
    X_test
)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"\nTest accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================
# CONFUSION MATRIX
# ============================================

matrix = confusion_matrix(
    y_test,
    test_predictions
)


print("\nConfusion Matrix:")
print(matrix)


# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ============================================
# PREDICTION DISTRIBUTION
# ============================================

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

prediction_counts = pd.Series(
    test_predictions
).value_counts().sort_index()


for value, count in prediction_counts.items():

    percentage = count / len(test_predictions) * 100

    print(
        f"Prediction {value}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ============================================
# PROBABILITY DISTRIBUTION
# ============================================

print("\n" + "=" * 60)
print("PROBABILITY DISTRIBUTION")
print("=" * 60)

print(
    f"\nMinimum probability: "
    f"{test_probabilities.min():.4f}"
)

print(
    f"Maximum probability: "
    f"{test_probabilities.max():.4f}"
)

print(
    f"Average probability: "
    f"{test_probabilities.mean():.4f}"
)


# ============================================
# RECENT PREDICTIONS
# ============================================

results = test.copy()

results["Prediction"] = test_predictions

results["Probability_Up"] = test_probabilities


print("\n" + "=" * 60)
print("RECENT PREDICTIONS")
print("=" * 60)


print(
    results[
        features
        + [
            "Target",
            "Prediction",
            "Probability_Up"
        ]
    ].tail(20).to_string()
)


# ============================================
# SAVE RESULTS
# ============================================

results.to_csv(
    "data/test_predictions_v1.csv"
)


print("\nSaved:")
print("data/test_predictions_v1.csv")


print("\n" + "=" * 60)
print("V1.0 ML MODEL COMPLETE")
print("=" * 60)