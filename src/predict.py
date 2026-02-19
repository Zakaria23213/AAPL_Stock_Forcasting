import numpy as np
import matplotlib.pyplot as plt
from data_loader import fetch_daily
from preprocessor import preprocess
from model import build_lstm

# ── 1. Load & preprocess ──────────────────────────────────────────
df = fetch_daily("AAPL")
X_train, X_test, y_train, y_test, scaler = preprocess(df)

# ── 2. Train model ────────────────────────────────────────────────
model = build_lstm()
model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_test, y_test), verbose=1)

# ── 3. Predict on test set ────────────────────────────────────────
predicted_scaled = model.predict(X_test)

# Scale back to real prices
predicted_prices = scaler.inverse_transform(predicted_scaled)
real_prices      = scaler.inverse_transform(y_test)

# ── 4. Forecast N days into the future ───────────────────────────
def forecast_future(model, df, scaler, sequence_len=60, days_ahead=10):
    data_scaled = scaler.transform(df[["close"]].values)
    last_sequence = data_scaled[-sequence_len:]  # last 60 days

    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days_ahead):
        x = current_seq.reshape(1, sequence_len, 1)
        next_val = model.predict(x, verbose=0)
        predictions.append(next_val[0, 0])
        # slide the window forward
        current_seq = np.append(current_seq[1:], next_val, axis=0)

    future_prices = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )
    return future_prices.flatten()

future = forecast_future(model, df, scaler, days_ahead=10)

# ── 5. Plot everything ────────────────────────────────────────────
plt.figure(figsize=(14, 6))

# Test set: real vs predicted
plt.plot(real_prices,      label="Real Price",      color="blue")
plt.plot(predicted_prices, label="Predicted Price", color="orange", linestyle="--")

# Future forecast appended after the test set
start = len(real_prices)
plt.plot(range(start, start + len(future)), future,
         label="Future Forecast (10 days)", color="green", linestyle="--", marker="o")

plt.title("AAPL Price — LSTM Forecast")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("forecast.png")  # saves the chart
plt.show()

print("\n # Next 10 days forecast:")
for i, price in enumerate(future, 1):
    print(f"  Day {i}: ${price:.2f}")