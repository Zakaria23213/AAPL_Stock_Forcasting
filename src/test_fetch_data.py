from data_loader import fetch_daily

df = fetch_daily("AAPL")

if df is not None:
    print(df.head())
    print(df.tail())