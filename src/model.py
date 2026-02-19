from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silences Info and Warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Stops the oneDNN floating-point message

def build_lstm(sequence_len=60):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_len, 1)),
        Dropout(0.2),   # randomly turns off 20% of neurons → prevents overfitting
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)        # output: 1 predicted price
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model