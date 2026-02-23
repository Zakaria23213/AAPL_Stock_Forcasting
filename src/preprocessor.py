import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, sequence_len=90):
    # Add moving average features
    df = df.copy()
    df["ma7"]  = df["close"].rolling(7).mean()   # 7-day average
    df["ma21"] = df["close"].rolling(21).mean()  # 21-day average
    df.dropna(inplace=True)

    features = ["close", "ma7", "ma21"]
    data = df[features].values  # shape: (days, 3)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_len, len(data_scaled)):
        X.append(data_scaled[i - sequence_len:i])  # shape: (90, 3)
        y.append(data_scaled[i, 0])                # predict close only

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler