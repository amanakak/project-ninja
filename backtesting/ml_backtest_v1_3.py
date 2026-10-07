import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# PROJECT NINJA - V1.3
# ML TRADING BACKTEST
# ============================================================

print("=" * 65)
print("PROJECT NINJA - V1.3 ML BACKTEST")
print("=" * 65)


# ============================================================
# CONFIG
# ============================================================

TICKER = "RELIANCE.NS"

TARGET_RETURN = 0.005

# Trading assumptions
TRANSACTION_COST = 0.001       # 0.10% per side
SLIPPAGE = 0.0005              # 0.05% per side

# Candidate probability thresholds
THRESHOLDS = [
    0.50,
    0.52,
    0.54,
    0.56,
    0.58,
    0.60
]

STARTING_CAPITAL = 100000


# ============================================================
# LOAD FEATURE DATA
# ============================================================

print("\nLoading datasets...")

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


print(f"Training:   {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test:       {len(test)}")


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


X_train = train[FEATURES]
y_train = train["Target"]

X_validation = validation[FEATURES]
y_validation = validation["Target"]

X_test = test[FEATURES]


# ============================================================
# TRAIN MODEL
# ============================================================

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


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

validation_probability = model.predict_proba(
    X_validation
)[:, 1]

test_probability = model.predict_proba(
    X_test
)[:, 1]


validation_data = validation.copy()

validation_data["Probability"] = validation_probability


test_data = test.copy()

test_data["Probability"] = test_probability


# ============================================================
# DOWNLOAD ACTUAL MARKET PRICES
# ============================================================

print("\nDownloading market prices...")

market = yf.Ticker(TICKER).history(
    period="15y"
)

market.index = pd.to_datetime(
    market.index
)

# Remove timezone so dates align cleanly
if market.index.tz is not None:
    market.index = market.index.tz_localize(None)


validation_data.index = pd.to_datetime(
    validation_data.index
).tz_localize(None)

test_data.index = pd.to_datetime(
    test_data.index
).tz_localize(None)


# ============================================================
# ALIGN MARKET DATA
# ============================================================

validation_data = validation_data.join(
    market[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ],
    how="left"
)

test_data = test_data.join(
    market[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ],
    how="left"
)


# ============================================================
# NEXT-DAY OPEN
# ============================================================

validation_data["Next_Open"] = (
    validation_data["Open"].shift(-1)
)

validation_data["Next_Close"] = (
    validation_data["Close"].shift(-1)
)

test_data["Next_Open"] = (
    test_data["Open"].shift(-1)
)

test_data["Next_Close"] = (
    test_data["Close"].shift(-1)
)


validation_data = validation_data.dropna(
    subset=[
        "Open",
        "Next_Open",
        "Next_Close"
    ]
)

test_data = test_data.dropna(
    subset=[
        "Open",
        "Next_Open",
        "Next_Close"
    ]
)


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def backtest(
    data,
    threshold,
    starting_capital=100000
):

    capital = starting_capital

    equity_curve = []

    trades = []

    position = False

    entry_price = None
    entry_date = None

    entry_capital = None

    for date, row in data.iterrows():

        probability = row["Probability"]

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        if not position and probability >= threshold:

            raw_entry = row["Next_Open"]

            entry_price = (
                raw_entry
                * (1 + SLIPPAGE)
                * (1 + TRANSACTION_COST)
            )

            entry_date = date

            entry_capital = capital

            position = True

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        elif position:

            raw_exit = row["Next_Open"]

            exit_price = (
                raw_exit
                * (1 - SLIPPAGE)
                * (1 - TRANSACTION_COST)
            )

            trade_return = (
                exit_price / entry_price - 1
            )

            capital = (
                entry_capital
                * (1 + trade_return)
            )

            trades.append({
                "Entry_Date": entry_date,
                "Exit_Date": date,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Return": trade_return,
                "P&L": capital - entry_capital
            })

            position = False

            entry_price = None
            entry_date = None
            entry_capital = None

        # ----------------------------------------------------
        # EQUITY
        # ----------------------------------------------------

        if position:

            current_price = row["Close"]

            unrealized_value = (
                entry_capital
                * current_price
                / entry_price
            )

            equity_curve.append(
                unrealized_value
            )

        else:

            equity_curve.append(
                capital
            )


    # ========================================================
    # FORCE CLOSE
    # ========================================================

    if position:

        last_row = data.iloc[-1]

        raw_exit = last_row["Close"]

        exit_price = (
            raw_exit
            * (1 - SLIPPAGE)
            * (1 - TRANSACTION_COST)
        )

        trade_return = (
            exit_price / entry_price - 1
        )

        capital = (
            entry_capital
            * (1 + trade_return)
        )

        trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": data.index[-1],
            "Entry_Price": entry_price,
            "Exit_Price": exit_price,
            "Return": trade_return,
            "P&L": capital - entry_capital
        })

        equity_curve[-1] = capital


    # ========================================================
    # STATISTICS
    # ========================================================

    equity = pd.Series(
        equity_curve,
        index=data.index
    )

    final_value = equity.iloc[-1]

    total_return = (
        final_value / starting_capital - 1
    )

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    )

    max_drawdown = drawdown.min()

    trades_df = pd.DataFrame(trades)

    if len(trades_df) > 0:

        winning_trades = trades_df[
            trades_df["P&L"] > 0
        ]

        losing_trades = trades_df[
            trades_df["P&L"] <= 0
        ]

        win_rate = (
            len(winning_trades)
            / len(trades_df)
        )

        gross_profit = (
            winning_trades["P&L"].sum()
        )

        gross_loss = abs(
            losing_trades["P&L"].sum()
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit / gross_loss
            )
        else:
            profit_factor = np.inf

    else:

        win_rate = 0
        profit_factor = 0

    return {
        "Final_Value": final_value,
        "Return": total_return,
        "Max_Drawdown": max_drawdown,
        "Trades": len(trades_df),
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Trades_Data": trades_df,
        "Equity": equity
    }


# ============================================================
# VALIDATION THRESHOLD TEST
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION THRESHOLD TEST")
print("=" * 65)

validation_results = []


for threshold in THRESHOLDS:

    result = backtest(
        validation_data,
        threshold,
        STARTING_CAPITAL
    )

    validation_results.append({
        "Threshold": threshold,
        "Return": result["Return"],
        "Max_Drawdown": result["Max_Drawdown"],
        "Trades": result["Trades"],
        "Win_Rate": result["Win_Rate"],
        "Profit_Factor": result["Profit_Factor"]
    })


validation_results_df = pd.DataFrame(
    validation_results
)


for _, row in validation_results_df.iterrows():

    print(
        f"\nThreshold: {row['Threshold']:.2f}"
    )

    print(
        f"Return: "
        f"{row['Return'] * 100:.2f}%"
    )

    print(
        f"Max Drawdown: "
        f"{row['Max_Drawdown'] * 100:.2f}%"
    )

    print(
        f"Trades: "
        f"{int(row['Trades'])}"
    )

    print(
        f"Win Rate: "
        f"{row['Win_Rate'] * 100:.2f}%"
    )

    print(
        f"Profit Factor: "
        f"{row['Profit_Factor']:.2f}"
    )


# ============================================================
# SELECT THRESHOLD
# ============================================================

# IMPORTANT:
# We choose the threshold using VALIDATION ONLY.
#
# We use return as the selection metric.
# The TEST set remains untouched until after selection.

best_row = validation_results_df.loc[
    validation_results_df["Return"].idxmax()
]

best_threshold = best_row["Threshold"]


print("\n" + "=" * 65)
print("SELECTED THRESHOLD")
print("=" * 65)

print(
    f"\nSelected threshold: "
    f"{best_threshold:.2f}"
)

print(
    "Selected using validation data only."
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 65)
print("FINAL OUT-OF-SAMPLE TEST")
print("=" * 65)


test_result = backtest(
    test_data,
    best_threshold,
    STARTING_CAPITAL
)


print(
    f"\nStarting Capital: "
    f"₹{STARTING_CAPITAL:,.2f}"
)

print(
    f"Final Value: "
    f"₹{test_result['Final_Value']:,.2f}"
)

print(
    f"P/L: "
    f"₹{test_result['Final_Value'] - STARTING_CAPITAL:,.2f}"
)

print(
    f"Return: "
    f"{test_result['Return'] * 100:.2f}%"
)

print(
    f"Max Drawdown: "
    f"{test_result['Max_Drawdown'] * 100:.2f}%"
)

print(
    f"Trades: "
    f"{test_result['Trades']}"
)

print(
    f"Win Rate: "
    f"{test_result['Win_Rate'] * 100:.2f}%"
)

print(
    f"Profit Factor: "
    f"{test_result['Profit_Factor']:.2f}"
)


# ============================================================
# BUY & HOLD
# ============================================================

buy_hold_start = test_data["Open"].iloc[0]

buy_hold_end = test_data["Close"].iloc[-1]

buy_hold_return = (
    buy_hold_end / buy_hold_start - 1
)


print("\n" + "=" * 65)
print("BUY & HOLD COMPARISON")
print("=" * 65)

print(
    f"\nBuy & Hold Return: "
    f"{buy_hold_return * 100:.2f}%"
)


# ============================================================
# TRADE LOG
# ============================================================

trades = test_result["Trades_Data"]


print("\n" + "=" * 65)
print("TEST TRADE LOG")
print("=" * 65)


if len(trades) == 0:

    print("\nNo trades were generated.")

else:

    for i, trade in trades.iterrows():

        print(
            f"\nTrade {i + 1}"
        )

        print(
            f"Entry: "
            f"{trade['Entry_Date']}"
        )

        print(
            f"Exit: "
            f"{trade['Exit_Date']}"
        )

        print(
            f"Entry Price: "
            f"₹{trade['Entry_Price']:.2f}"
        )

        print(
            f"Exit Price: "
            f"₹{trade['Exit_Price']:.2f}"
        )

        print(
            f"Return: "
            f"{trade['Return'] * 100:.2f}%"
        )

        print(
            f"P&L: "
            f"₹{trade['P&L']:,.2f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

validation_results_df.to_csv(
    "data/ml_validation_thresholds_v1_3.csv",
    index=False
)

trades.to_csv(
    "data/ml_test_trades_v1_3.csv",
    index=False
)

test_result["Equity"].to_csv(
    "data/ml_test_equity_v1_3.csv"
)


print("\n" + "=" * 65)
print("FILES SAVED")
print("=" * 65)

print(
    "\ndata/ml_validation_thresholds_v1_3.csv"
)

print(
    "data/ml_test_trades_v1_3.csv"
)

print(
    "data/ml_test_equity_v1_3.csv"
)


print("\n" + "=" * 65)
print("V1.3 COMPLETE")
print("=" * 65)