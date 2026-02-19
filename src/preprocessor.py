import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, feature_col="close", sequence_len=60):
    """
    feature_col  : which column to predict (close price)
    sequence_len : how many past days the LSTM looks at (60 days → predict next N)
    """
    # Keep only the closing price
    data = df[[feature_col]].values  # shape: (num_days, 1)

    # Scale values to range [0, 1] — LSTMs train better on small numbers
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Build sequences: X = past 60 days, y = next day
    X, y = [], []
    for i in range(sequence_len, len(data_scaled)):
        X.append(data_scaled[i - sequence_len:i])  # 60 days of input
        y.append(data_scaled[i])                   # 1 day to predict

    X = np.array(X)  # shape: (samples, 60, 1)
    y = np.array(y)  # shape: (samples, 1)

    # Split: 80% train, 20% test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler