import yfinance as yf

def fetch_daily(symbol):
    df = yf.download(symbol, period="2y", interval="1d", auto_adjust=True)
    df.columns = df.columns.get_level_values(0).str.lower()  # flatten MultiIndex
    return df