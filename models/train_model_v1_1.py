import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)


# ============================================
# PROJECT NINJA - V1.1
# BALANCED LOGISTIC REGRESSION
# ============================================

print("=" * 60)
print("PROJECT NINJA - V1.1")
print("BALANCED LOGISTIC REGRESSION")
print("=" * 60)


# ============================================
# LOAD DATA
# ============================================

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


print("\nTraining samples:", len(train))
print("Validation samples:", len(validation))
print("Testing samples:", len(test))


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
# MODEL
# ============================================

print("\nTraining balanced Logistic Regression...")


model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


model.fit(
    X_train,
    y_train
)


print("Training complete.")


# ============================================
# VALIDATION
# ============================================

validation_predictions = model.predict(
    X_validation
)

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]


validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_precision = precision_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_recall = recall_score(
    y_validation,
    validation_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Precision for class 1: "
    f"{validation_precision * 100:.2f}%"
)

print(
    f"Recall for class 1: "
    f"{validation_recall * 100:.2f}%"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        validation_predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_predictions,
        zero_division=0
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

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Precision for class 1: "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Recall for class 1: "
    f"{test_recall * 100:.2f}%"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ============================================
# PREDICTION DISTRIBUTION
# ============================================

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

counts = pd.Series(
    test_predictions
).value_counts().sort_index()

for prediction, count in counts.items():

    print(
        f"Prediction {prediction}: "
        f"{count} "
        f"({count / len(test_predictions) * 100:.2f}%)"
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
# SAVE
# ============================================

results.to_csv(
    "data/test_predictions_v1_1.csv"
)

print(
    "\nSaved: "
    "data/test_predictions_v1_1.csv"
)


print("\n" + "=" * 60)
print("V1.1 COMPLETE")
print("=" * 60)