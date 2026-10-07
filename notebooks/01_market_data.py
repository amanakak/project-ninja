import yfinance as yf

ticker = yf.Ticker("RELIANCE.NS")

data = ticker.history(period="1mo")

print(data)