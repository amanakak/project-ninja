import pandas as pd
import numpy as np


# ============================================
# PROJECT NINJA - V1.2
# PROBABILITY ANALYSIS
# ============================================

print("=" * 60)
print("PROJECT NINJA - V1.2")
print("PROBABILITY / FORWARD RETURN ANALYSIS")
print("=" * 60)


# ============================================
# LOAD PREDICTIONS
# ============================================

data = pd.read_csv(
    "data/test_predictions_v1_1.csv",
    index_col=0
)

print(f"\nTest samples: {len(data)}")


# ============================================
# RECREATE NEXT-DAY RETURN
# ============================================

# We need the actual next-day return.
#
# The prediction was made using information
# available at the current day's close.
#
# Future return = next close / current close - 1

import yfinance as yf

market = yf.Ticker("RELIANCE.NS").history(
    start="2023-11-16",
    end="2026-09-19"
)


market["Future_Return"] = (
    market["Close"].shift(-1)
    / market["Close"]
    - 1
)


# Align dates

market.index = market.index.normalize()
data.index = pd.to_datetime(data.index).normalize()


data = data.join(
    market[["Future_Return"]],
    how="left"
)


data = data.dropna(
    subset=["Future_Return"]
)


# ============================================
# PROBABILITY BUCKETS
# ============================================

bins = [
    0.00,
    0.45,
    0.50,
    0.55,
    0.60,
    1.00
]

labels = [
    "<0.45",
    "0.45-0.50",
    "0.50-0.55",
    "0.55-0.60",
    ">0.60"
]


data["Probability_Bucket"] = pd.cut(
    data["Probability_Up"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# ============================================
# ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("PROBABILITY BUCKET ANALYSIS")
print("=" * 60)


results = []


for bucket in labels:

    subset = data[
        data["Probability_Bucket"] == bucket
    ]

    if len(subset) == 0:
        continue

    avg_return = (
        subset["Future_Return"].mean()
    )

    positive_rate = (
        subset["Future_Return"] > 0.005
    ).mean()

    average_winning_return = (
        subset.loc[
            subset["Future_Return"] > 0,
            "Future_Return"
        ].mean()
    )

    average_return_after_costs = (
        avg_return - 0.0015
    )

    total_return = (
        (1 + subset["Future_Return"]).prod() - 1
    )

    results.append({
        "Bucket": bucket,
        "Samples": len(subset),
        "Avg_Return": avg_return,
        "Pct_Above_0.5": positive_rate,
        "Avg_Winning_Return": average_winning_return,
        "Avg_Return_After_Costs": average_return_after_costs,
        "Compound_Return": total_return
    })


results_df = pd.DataFrame(results)


# ============================================
# DISPLAY
# ============================================

for _, row in results_df.iterrows():

    print(
        f"\n{row['Bucket']}"
    )

    print(
        f"Samples: "
        f"{int(row['Samples'])}"
    )

    print(
        f"Average next-day return: "
        f"{row['Avg_Return'] * 100:.3f}%"
    )

    print(
        f"Days above +0.5%: "
        f"{row['Pct_Above_0.5'] * 100:.2f}%"
    )

    print(
        f"Average winning return: "
        f"{row['Avg_Winning_Return'] * 100:.3f}%"
    )

    print(
        f"Average return after assumed costs: "
        f"{row['Avg_Return_After_Costs'] * 100:.3f}%"
    )

    print(
        f"Compound return: "
        f"{row['Compound_Return'] * 100:.2f}%"
    )


# ============================================
# CORRELATION
# ============================================

correlation = data[
    [
        "Probability_Up",
        "Future_Return"
    ]
].corr().iloc[0, 1]


print("\n" + "=" * 60)
print("PROBABILITY / RETURN CORRELATION")
print("=" * 60)

print(
    f"\nCorrelation: {correlation:.4f}"
)


# ============================================
# TOP SIGNALS
# ============================================

print("\n" + "=" * 60)
print("TOP 20 HIGHEST-PROBABILITY SIGNALS")
print("=" * 60)


top = data.sort_values(
    "Probability_Up",
    ascending=False
).head(20)


print(
    top[
        [
            "Probability_Up",
            "Future_Return",
            "Target"
        ]
    ].to_string()
)


# ============================================
# SAVE
# ============================================

results_df.to_csv(
    "data/probability_analysis_v1_2.csv",
    index=False
)


print("\nSaved:")
print("data/probability_analysis_v1_2.csv")


print("\n" + "=" * 60)
print("V1.2 COMPLETE")
print("=" * 60)