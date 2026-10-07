import yfinance as yf
import pandas as pd


# ==========================================
# PROJECT NINJA V0.8
# FEATURE ENGINE + DATASET BUILDER
# ==========================================


# ==========================================
# 1. DOWNLOAD MARKET DATA
# ==========================================

print()
print("Downloading market data...")

data = yf.Ticker("RELIANCE.NS").history(
    period="15y"
)

print(
    f"Downloaded {len(data)} rows."
)


# ==========================================
# 2. PRICE FEATURES
# ==========================================

# 1-day return
data["Return_1D"] = (
    data["Close"].pct_change()
)

# 5-day return
data["Return_5D"] = (
    data["Close"].pct_change(5)
)

# 20-day return
data["Return_20D"] = (
    data["Close"].pct_change(20)
)


# ==========================================
# 3. MOVING AVERAGES
# ==========================================

data["MA20"] = (
    data["Close"].rolling(20).mean()
)

data["MA50"] = (
    data["Close"].rolling(50).mean()
)

data["MA200"] = (
    data["Close"].rolling(200).mean()
)


# ==========================================
# 4. DISTANCE FROM MOVING AVERAGES
# ==========================================

data["Distance_MA20"] = (
    data["Close"] /
    data["MA20"] - 1
)

data["Distance_MA50"] = (
    data["Close"] /
    data["MA50"] - 1
)

data["Distance_MA200"] = (
    data["Close"] /
    data["MA200"] - 1
)


# ==========================================
# 5. RSI
# ==========================================

delta = data["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

data["RSI"] = (
    100 -
    (100 / (1 + rs))
)


# ==========================================
# 6. VOLATILITY
# ==========================================

data["Volatility_20D"] = (
    data["Return_1D"]
    .rolling(20)
    .std()
)


# ==========================================
# 7. VOLUME FEATURES
# ==========================================

data["Volume_MA20"] = (
    data["Volume"]
    .rolling(20)
    .mean()
)

data["Volume_Ratio"] = (
    data["Volume"] /
    data["Volume_MA20"]
)


# Remove invalid zero-volume observations

data.loc[
    data["Volume"] <= 0,
    "Volume_Ratio"
] = pd.NA


# ==========================================
# 8. FUTURE RETURN
# ==========================================

# What happens on the NEXT trading day?

data["Future_Return"] = (
    data["Close"].shift(-1) /
    data["Close"]
) - 1


# Remove the final row because there is
# no future trading day available.

data = data.dropna(
    subset=["Future_Return"]
)


# ==========================================
# 9. TARGET
# ==========================================

# 1 = next trading day goes UP
# 0 = next trading day does NOT go UP

data["Target"] = (
    data["Future_Return"] > 0
).astype(int)


# ==========================================
# 10. FEATURES USED BY ML
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
# 11. CREATE CLEAN DATASET
# ==========================================

dataset = data[
    features + ["Target"]
].dropna()


# ==========================================
# 12. CHRONOLOGICAL SPLIT
# ==========================================

# 60% → Training
# 20% → Validation
# 20% → Testing

train_end = int(
    len(dataset) * 0.60
)

validation_end = int(
    len(dataset) * 0.80
)


train = dataset.iloc[
    :train_end
]

validation = dataset.iloc[
    train_end:validation_end
]

test = dataset.iloc[
    validation_end:
]


# ==========================================
# 13. SAVE DATASETS
# ==========================================

train.to_csv(
    "data/train.csv"
)

validation.to_csv(
    "data/validation.csv"
)

test.to_csv(
    "data/test.csv"
)


# ==========================================
# 14. PRINT INFORMATION
# ==========================================

print()
print("========================================")
print("        PROJECT NINJA V0.8")
print("        DATASET BUILDER")
print("========================================")


# ==========================================
# DATASET SIZE
# ==========================================

print()
print("Dataset sizes:")
print("----------------------------------------")

print(
    f"Total samples:        {len(dataset)}"
)

print(
    f"Training samples:     {len(train)}"
)

print(
    f"Validation samples:   {len(validation)}"
)

print(
    f"Testing samples:      {len(test)}"
)


# ==========================================
# TRAINING PERIOD
# ==========================================

print()
print("Training period:")
print("----------------------------------------")

print(
    train.index[0],
    "→",
    train.index[-1]
)


# ==========================================
# VALIDATION PERIOD
# ==========================================

print()
print("Validation period:")
print("----------------------------------------")

print(
    validation.index[0],
    "→",
    validation.index[-1]
)


# ==========================================
# TEST PERIOD
# ==========================================

print()
print("Testing period:")
print("----------------------------------------")

print(
    test.index[0],
    "→",
    test.index[-1]
)


# ==========================================
# TARGET DISTRIBUTION
# ==========================================

print()
print("Target distribution:")
print("----------------------------------------")

print("TRAIN:")

print(
    train["Target"]
    .value_counts()
    .sort_index()
)

print()

print("VALIDATION:")

print(
    validation["Target"]
    .value_counts()
    .sort_index()
)

print()

print("TEST:")

print(
    test["Target"]
    .value_counts()
    .sort_index()
)


# ==========================================
# TARGET PERCENTAGES
# ==========================================

print()
print("Target percentages:")
print("----------------------------------------")

print("TRAIN:")

print(
    train["Target"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print()

print("VALIDATION:")

print(
    validation["Target"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print()

print("TEST:")

print(
    test["Target"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


# ==========================================
# FEATURES
# ==========================================

print()
print("Features:")
print("----------------------------------------")

for feature in features:

    print(
        f"- {feature}"
    )


# ==========================================
# RECENT DATA
# ==========================================

print()
print("Recent dataset:")
print("----------------------------------------")

print(
    dataset.tail(10)
)


# ==========================================
# FINISHED
# ==========================================

print()
print("========================================")
print("Dataset creation complete.")
print("========================================")

print()
print("Files created:")

print(
    "data/train.csv"
)

print(
    "data/validation.csv"
)

print(
    "data/test.csv"
)

print()