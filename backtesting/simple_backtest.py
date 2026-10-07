import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PROJECT NINJA V0.6
# PROPER EXECUTION TIMING
# ==========================================

STARTING_CASH = 100000

TRANSACTION_COST = 0.0010   # 0.10%
SLIPPAGE = 0.0005           # 0.05%


# ==========================================
# 1. GET DATA
# ==========================================

data = yf.Ticker("RELIANCE.NS").history(period="1y")

data = data.dropna(subset=["Open", "Close"])


# ==========================================
# 2. INDICATORS
# ==========================================

data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()


# RSI
delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))


# ==========================================
# 3. SIGNAL
# ==========================================

data["Signal"] = 0

buy_condition = (
    (data["MA20"] > data["MA50"]) &
    (data["RSI"] > 50)
)

sell_condition = (
    (data["MA20"] < data["MA50"]) |
    (data["RSI"] < 40)
)

data.loc[buy_condition, "Signal"] = 1
data.loc[sell_condition, "Signal"] = -1


# ==========================================
# 4. ACCOUNT
# ==========================================

cash = STARTING_CASH
shares = 0

portfolio_values = []

trades = []

bars_in_market = 0


# ==========================================
# 5. BACKTEST
# ==========================================

# Start from the second trading day.
# The previous day's signal determines
# today's execution.

for i in range(1, len(data)):

    date = data.index[i]
    row = data.iloc[i]

    previous_row = data.iloc[i - 1]

    open_price = float(row["Open"])
    close_price = float(row["Close"])

    signal = int(previous_row["Signal"])


    # ======================================
    # BUY AT TODAY'S OPEN
    # ======================================

    if signal == 1 and shares == 0:

        execution_price = (
            open_price *
            (1 + SLIPPAGE)
        )

        shares_to_buy = int(
            cash /
            (
                execution_price *
                (1 + TRANSACTION_COST)
            )
        )

        if shares_to_buy > 0:

            trade_value = (
                shares_to_buy *
                execution_price
            )

            fee = (
                trade_value *
                TRANSACTION_COST
            )

            total_cost = (
                trade_value +
                fee
            )

            cash -= total_cost

            shares = shares_to_buy

            trades.append({
                "date": date,
                "type": "BUY",
                "shares": shares,
                "price": execution_price,
                "fee": fee
            })


    # ======================================
    # SELL AT TODAY'S OPEN
    # ======================================

    elif signal == -1 and shares > 0:

        execution_price = (
            open_price *
            (1 - SLIPPAGE)
        )

        trade_value = (
            shares *
            execution_price
        )

        fee = (
            trade_value *
            TRANSACTION_COST
        )

        cash += (
            trade_value -
            fee
        )

        trades.append({
            "date": date,
            "type": "SELL",
            "shares": shares,
            "price": execution_price,
            "fee": fee
        })

        shares = 0


    # ======================================
    # PORTFOLIO VALUE
    # ======================================

    portfolio_value = (
        cash +
        shares * close_price
    )

    portfolio_values.append(portfolio_value)


    if shares > 0:
        bars_in_market += 1


# ==========================================
# 6. FORCE CLOSE AT FINAL CLOSE
# ==========================================

if shares > 0:

    final_date = data.index[-1]

    final_close = float(
        data["Close"].iloc[-1]
    )

    execution_price = (
        final_close *
        (1 - SLIPPAGE)
    )

    trade_value = (
        shares *
        execution_price
    )

    fee = (
        trade_value *
        TRANSACTION_COST
    )

    cash += (
        trade_value -
        fee
    )

    trades.append({
        "date": final_date,
        "type": "SELL",
        "shares": shares,
        "price": execution_price,
        "fee": fee
    })

    shares = 0

    # Update final portfolio value
    portfolio_values[-1] = cash


# ==========================================
# 7. EQUITY CURVE
# ==========================================

# Portfolio values correspond to data from
# the second trading day onward.

equity_dates = data.index[1:]

equity = pd.DataFrame(
    index=equity_dates
)

equity["Portfolio"] = portfolio_values


# ==========================================
# 8. PERFORMANCE
# ==========================================

final_value = cash

profit = (
    final_value -
    STARTING_CASH
)

total_return = (
    profit /
    STARTING_CASH
) * 100


# ==========================================
# 9. BUY & HOLD
# ==========================================

first_close = float(
    data["Close"].iloc[0]
)

last_close = float(
    data["Close"].iloc[-1]
)

buy_hold_return = (
    (last_close / first_close) - 1
) * 100


# ==========================================
# 10. TRADE PAIRS
# ==========================================

completed_trades = []

for i in range(0, len(trades) - 1, 2):

    buy = trades[i]
    sell = trades[i + 1]

    if (
        buy["type"] == "BUY"
        and sell["type"] == "SELL"
    ):

        buy_cost = (
            buy["shares"] *
            buy["price"] +
            buy["fee"]
        )

        sell_revenue = (
            sell["shares"] *
            sell["price"] -
            sell["fee"]
        )

        trade_profit = (
            sell_revenue -
            buy_cost
        )

        completed_trades.append({
            "buy_date": buy["date"],
            "sell_date": sell["date"],
            "buy_price": buy["price"],
            "sell_price": sell["price"],
            "shares": buy["shares"],
            "profit": trade_profit
        })


# ==========================================
# 11. TRADE STATISTICS
# ==========================================

trade_profits = [
    trade["profit"]
    for trade in completed_trades
]

winning_trades = [
    x for x in trade_profits
    if x > 0
]

losing_trades = [
    x for x in trade_profits
    if x < 0
]

total_trades = len(trade_profits)


if total_trades > 0:

    win_rate = (
        len(winning_trades) /
        total_trades
    ) * 100

else:

    win_rate = 0


if winning_trades:

    average_win = (
        sum(winning_trades) /
        len(winning_trades)
    )

else:

    average_win = 0


if losing_trades:

    average_loss = (
        sum(losing_trades) /
        len(losing_trades)
    )

else:

    average_loss = 0


gross_profit = sum(winning_trades)

gross_loss = abs(
    sum(losing_trades)
)


if gross_loss > 0:

    profit_factor = (
        gross_profit /
        gross_loss
    )

else:

    profit_factor = 0


if trade_profits:

    largest_win = max(
        trade_profits
    )

    largest_loss = min(
        trade_profits
    )

else:

    largest_win = 0
    largest_loss = 0


# ==========================================
# 12. MAX DRAWDOWN
# ==========================================

equity["Peak"] = (
    equity["Portfolio"].cummax()
)

equity["Drawdown"] = (
    equity["Portfolio"] -
    equity["Peak"]
) / equity["Peak"]

max_drawdown = (
    equity["Drawdown"].min()
) * 100


# ==========================================
# 13. TIME IN MARKET
# ==========================================

total_bars = len(equity)

if total_bars > 0:

    time_in_market = (
        bars_in_market /
        total_bars
    ) * 100

else:

    time_in_market = 0


# ==========================================
# 14. PRINT RESULTS
# ==========================================

print()
print("========================================")
print("        PROJECT NINJA V0.6")
print("========================================")

print()
print("BACKTEST SETTINGS")
print("----------------------------------------")

print(
    f"Starting capital:     ₹{STARTING_CASH:,.2f}"
)

print(
    f"Transaction cost:     "
    f"{TRANSACTION_COST * 100:.2f}%"
)

print(
    f"Slippage:             "
    f"{SLIPPAGE * 100:.2f}%"
)

print()
print("PERFORMANCE")
print("----------------------------------------")

print(
    f"Final value:           "
    f"₹{final_value:,.2f}"
)

print(
    f"Profit/Loss:           "
    f"₹{profit:,.2f}"
)

print(
    f"Strategy return:       "
    f"{total_return:.2f}%"
)

print(
    f"Buy & Hold return:     "
    f"{buy_hold_return:.2f}%"
)

print()
print("TRADE STATISTICS")
print("----------------------------------------")

print(
    f"Completed trades:      "
    f"{total_trades}"
)

print(
    f"Winning trades:        "
    f"{len(winning_trades)}"
)

print(
    f"Losing trades:         "
    f"{len(losing_trades)}"
)

print(
    f"Win rate:              "
    f"{win_rate:.2f}%"
)

print()
print("TRADE QUALITY")
print("----------------------------------------")

print(
    f"Average winning trade: "
    f"₹{average_win:,.2f}"
)

print(
    f"Average losing trade:  "
    f"₹{average_loss:,.2f}"
)

print(
    f"Profit factor:         "
    f"{profit_factor:.2f}"
)

print(
    f"Largest win:           "
    f"₹{largest_win:,.2f}"
)

print(
    f"Largest loss:          "
    f"₹{largest_loss:,.2f}"
)

print()
print("RISK")
print("----------------------------------------")

print(
    f"Maximum drawdown:      "
    f"{max_drawdown:.2f}%"
)

print(
    f"Time in market:        "
    f"{time_in_market:.2f}%"
)

print()
print("========================================")


# ==========================================
# 15. TRADE LOG
# ==========================================

print()
print("TRADE LOG")
print("----------------------------------------")

if completed_trades:

    for number, trade in enumerate(
        completed_trades,
        start=1
    ):

        print(
            f"{number}. "
            f"{trade['buy_date'].date()} BUY "
            f"@ ₹{trade['buy_price']:.2f} → "
            f"{trade['sell_date'].date()} SELL "
            f"@ ₹{trade['sell_price']:.2f} | "
            f"P/L: ₹{trade['profit']:,.2f}"
        )

else:

    print("No completed trades.")


# ==========================================
# 16. RECENT SIGNALS
# ==========================================

print()
print("RECENT MARKET SIGNALS")
print("----------------------------------------")

print(
    data[
        [
            "Close",
            "Open",
            "MA20",
            "MA50",
            "RSI",
            "Signal"
        ]
    ].tail(10)
)


# ==========================================
# 17. PLOT
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    equity.index,
    equity["Portfolio"],
    label="Project Ninja V0.6"
)

plt.title(
    "Project Ninja V0.6 - Portfolio Value"
)

plt.xlabel("Date")
plt.ylabel("Portfolio Value (₹)")

plt.legend()

plt.grid()

plt.show()