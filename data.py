import yfinance as yf
import pandas as pd

tickers = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN"]

start_date = "2020-01-01"
end_date = "2024-03-19"

data = yf.download(tickers, start=start_date, end=end_date)['Close']

data.reset_index(inplace=True)

data.rename(columns={"Date": "date"}, inplace=True)

print(data.head())

data.to_csv("nasdaq_stock_prices.csv", index=False)