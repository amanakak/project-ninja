import yfinance as yf
import pandas as pd

# Get 1 year of Reliance data
data = yf.Ticker("RELIANCE.NS").history(period="1y")

# Calculate the 20-day moving average
data["MA20"] = data["Close"].rolling(window=20).mean()

# Create a trading signal
data["Signal"] = 0

data.loc[data["Close"] > data["MA20"], "Signal"] = 1
data.loc[data["Close"] < data["MA20"], "Signal"] = -1

# Show the latest data
print(data[["Close", "MA20", "Signal"]].tail(20))