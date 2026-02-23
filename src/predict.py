import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from data_loader import fetch_daily
from preprocessor import preprocess
from model import build_lstm

# ── 1. Load & preprocess ──────────────────────────────────────────
df = fetch_daily("AAPL")
X_train, X_test, y_train, y_test, scaler = preprocess(df)

# ── 2. Train model ────────────────────────────────────────────────
model = build_lstm(sequence_len=90, n_features=3)
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_data=(X_test, y_test), verbose=1)

# ── 3. Predict on test set ────────────────────────────────────────
predicted_scaled = model.predict(X_test)

# Scaler has 3 features (close, ma7, ma21) so we must pad with zeros
# before inverse transforming, then grab column 0 (close price only)
def inverse_close(scaler, values):
    dummy = np.zeros((len(values), 3))
    dummy[:, 0] = values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

predicted_prices = inverse_close(scaler, predicted_scaled)  # ← correct
real_prices      = inverse_close(scaler, y_test)            # ← correct

# ── 4. Forecast N days into the future ───────────────────────────
def forecast_future(model, df, scaler, sequence_len=90, days_ahead=10):
    df = df.copy()
    df["ma7"]  = df["close"].rolling(7).mean()
    df["ma21"] = df["close"].rolling(21).mean()
    df.dropna(inplace=True)

    data_scaled = scaler.transform(df[["close", "ma7", "ma21"]].values)
    last_sequence = data_scaled[-sequence_len:]

    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days_ahead):
        x = current_seq.reshape(1, sequence_len, 3)
        next_val = model.predict(x, verbose=0)[0, 0]
        predictions.append(next_val)

        new_row = np.array([[next_val, current_seq[-1, 1], current_seq[-1, 2]]])
        current_seq = np.append(current_seq[1:], new_row, axis=0)

    dummy = np.zeros((len(predictions), 3))
    dummy[:, 0] = predictions
    future_prices = scaler.inverse_transform(dummy)[:, 0]
    return future_prices

future = forecast_future(model, df, scaler, sequence_len=90, days_ahead=10)

# ── 5. Plot everything ────────────────────────────────────────────
plt.figure(figsize=(14, 6))

plt.plot(real_prices,      label="Real Price",      color="blue")
plt.plot(predicted_prices, label="Predicted Price", color="orange", linestyle="--")

start = len(real_prices)
plt.plot(range(start, start + len(future)), future,
         label="Future Forecast (10 days)", color="green", linestyle="--", marker="o")

plt.title("AAPL Price — LSTM Forecast")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("forecast.png")
plt.show()

print("\n📈 Next 10 days forecast:")
for i, price in enumerate(future, 1):
    print(f"  Day {i}: ${price:.2f}")