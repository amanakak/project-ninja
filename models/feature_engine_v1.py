import yfinance as yf
import pandas as pd


# ============================================
# PROJECT NINJA - V1.0 FEATURE ENGINE
# ============================================

TICKER = "RELIANCE.NS"
HISTORY = "15y"

# Target threshold
TARGET_RETURN = 0.005   # +0.5%


print("=" * 60)
print("PROJECT NINJA - V1.0 FEATURE ENGINE")
print("=" * 60)

print("\nDownloading market data...")

data = yf.Ticker(TICKER).history(period=HISTORY)

print(f"Downloaded {len(data)} rows")


# ============================================
# FEATURE ENGINEERING
# ============================================

# 1-day return
data["Return_1D"] = data["Close"].pct_change()

# 5-day return
data["Return_5D"] = data["Close"].pct_change(5)

# 20-day return
data["Return_20D"] = data["Close"].pct_change(20)


# Moving averages
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()


# Distance from moving averages
data["Distance_MA20"] = (
    data["Close"] / data["MA20"] - 1
)

data["Distance_MA50"] = (
    data["Close"] / data["MA50"] - 1
)

data["Distance_MA200"] = (
    data["Close"] / data["MA200"] - 1
)


# ============================================
# RSI
# ============================================

delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))


# ============================================
# VOLATILITY
# ============================================

data["Volatility_20D"] = (
    data["Return_1D"].rolling(20).std()
)


# ============================================
# VOLUME
# ============================================

data["Volume_MA20"] = (
    data["Volume"].rolling(20).mean()
)

data["Volume_Ratio"] = (
    data["Volume"] / data["Volume_MA20"]
)

# Remove invalid zero-volume observations
data.loc[data["Volume"] <= 0, "Volume_Ratio"] = pd.NA


# ============================================
# TARGET
# ============================================

# Tomorrow's return
data["Future_Return"] = (
    data["Close"].shift(-1) / data["Close"] - 1
)

# Remove final row because tomorrow's return doesn't exist
data = data.dropna(subset=["Future_Return"])


# V1.0 TARGET
#
# 1 = tomorrow rises more than 0.5%
# 0 = tomorrow does not rise more than 0.5%

data["Target"] = (
    data["Future_Return"] > TARGET_RETURN
).astype(int)


# ============================================
# SELECT FEATURES
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


dataset = data[features + ["Target"]].dropna()


# ============================================
# CHRONOLOGICAL SPLIT
# ============================================

train_end = int(len(dataset) * 0.60)
validation_end = int(len(dataset) * 0.80)

train = dataset.iloc[:train_end]
validation = dataset.iloc[train_end:validation_end]
test = dataset.iloc[validation_end:]


# ============================================
# SAVE DATASETS
# ============================================

train.to_csv("data/train_v1.csv")
validation.to_csv("data/validation_v1.csv")
test.to_csv("data/test_v1.csv")


# ============================================
# REPORT
# ============================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"\nTotal usable samples: {len(dataset)}")

print(f"\nTraining samples: {len(train)}")
print(f"Validation samples: {len(validation)}")
print(f"Testing samples: {len(test)}")


print("\nTraining period:")
print(train.index.min(), "->", train.index.max())

print("\nValidation period:")
print(validation.index.min(), "->", validation.index.max())

print("\nTesting period:")
print(test.index.min(), "->", test.index.max())


# ============================================
# TARGET DISTRIBUTION
# ============================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    counts = df["Target"].value_counts().sort_index()

    print(f"\n{name}")

    print(
        f"Target 0: {counts.get(0, 0)} "
        f"({counts.get(0, 0) / len(df) * 100:.2f}%)"
    )

    print(
        f"Target 1: {counts.get(1, 0)} "
        f"({counts.get(1, 0) / len(df) * 100:.2f}%)"
    )


# ============================================
# FEATURES
# ============================================

print("\n" + "=" * 60)
print("FEATURES")
print("=" * 60)

for feature in features:
    print("-", feature)


# ============================================
# RECENT DATA
# ============================================

print("\n" + "=" * 60)
print("RECENT DATA")
print("=" * 60)

print(
    dataset.tail(10).to_string()
)


print("\n" + "=" * 60)
print("V1.0 FEATURE ENGINE COMPLETE")
print("=" * 60)