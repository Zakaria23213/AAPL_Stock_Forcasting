import yfinance as yf

def fetch_daily(symbol):
    df = yf.download(symbol, period="5y", interval="1d", auto_adjust=True)  # ← change here
    df.columns = df.columns.get_level_values(0).str.lower()
    return df