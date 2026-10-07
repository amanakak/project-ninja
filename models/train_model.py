import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# PROJECT NINJA V0.9
# FIRST ML MODEL
# ==========================================


# ==========================================
# 1. LOAD DATASETS
# ==========================================

train = pd.read_csv(
    "data/train.csv",
    index_col=0
)

validation = pd.read_csv(
    "data/validation.csv",
    index_col=0
)

test = pd.read_csv(
    "data/test.csv",
    index_col=0
)


# ==========================================
# 2. FEATURES
# ==========================================

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


# ==========================================
# 3. SPLIT FEATURES / TARGET
# ==========================================

X_train = train[features]
y_train = train["Target"]

X_validation = validation[features]
y_validation = validation["Target"]

X_test = test[features]
y_test = test["Target"]


# ==========================================
# 4. BASELINE 1
# ALWAYS PREDICT UP
# ==========================================

always_up = [1] * len(y_test)

always_up_accuracy = accuracy_score(
    y_test,
    always_up
)


# ==========================================
# 5. BASELINE 2
# ALWAYS PREDICT DOWN
# ==========================================

always_down = [0] * len(y_test)

always_down_accuracy = accuracy_score(
    y_test,
    always_down
)


# ==========================================
# 6. CREATE LOGISTIC REGRESSION
# ==========================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# ==========================================
# 7. TRAIN
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# 8. VALIDATION
# ==========================================

validation_predictions = model.predict(
    X_validation
)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)


# ==========================================
# 9. FINAL TEST PREDICTIONS
# ==========================================

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


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    y_test,
    test_predictions
)


# ==========================================
# 11. PRINT RESULTS
# ==========================================

print()
print("========================================")
print("        PROJECT NINJA V0.9")
print("        FIRST ML MODEL")
print("========================================")

print()

print("DATASET")
print("----------------------------------------")

print(
    f"Training samples:     {len(X_train)}"
)

print(
    f"Validation samples:   {len(X_validation)}"
)

print(
    f"Testing samples:      {len(X_test)}"
)


print()

print("BASELINES")
print("----------------------------------------")

print(
    f"Always UP accuracy:   "
    f"{always_up_accuracy * 100:.2f}%"
)

print(
    f"Always DOWN accuracy: "
    f"{always_down_accuracy * 100:.2f}%"
)


print()

print("LOGISTIC REGRESSION")
print("----------------------------------------")

print(
    f"Validation accuracy:  "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy:        "
    f"{test_accuracy * 100:.2f}%"
)


print()

print("CONFUSION MATRIX")
print("----------------------------------------")

print(matrix)


print()

print("CLASSIFICATION REPORT")
print("----------------------------------------")

print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ==========================================
# 12. SAMPLE PREDICTIONS
# ==========================================

print()

print("RECENT PREDICTIONS")
print("----------------------------------------")

results = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": test_predictions,

    "Probability_UP":
        test_probabilities

}, index=test.index)


print(
    results.tail(20)
)


print()
print("========================================")