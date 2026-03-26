import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# 1. LOAD DATA
# =========================
print("Loading datasets...")

train_data = pd.read_csv(r"C:\Users\Keert\OneDrive\Desktop\weather prediction\archive\DailyDelhiClimateTrain.csv")
test_data = pd.read_csv(r"C:\Users\Keert\OneDrive\Desktop\weather prediction\archive\DailyDelhiClimateTest.csv")

print("Train + Test loaded!")

# Use multiple features
train_features = train_data[['meantemp', 'humidity', 'wind_speed']]
test_features = test_data[['meantemp', 'humidity', 'wind_speed']]

# =========================
# 2. NORMALIZE (fit only on train!)
# =========================
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

# =========================
# 3. CREATE SEQUENCES
# =========================
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # predict temp
    return np.array(X), np.array(y)

seq_length = 10

X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# =========================
# 4. BUILD MODEL
# =========================
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_length, 3)),
    Dropout(0.2),

    LSTM(100),
    Dropout(0.2),

    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("Model ready!")

# =========================
# 5. TRAIN MODEL
# =========================
print("Training started...")

model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)

print("Training completed!")

# =========================
# 6. PREDICT ON TEST DATA
# =========================
print("Predicting on test data...")

predictions = model.predict(X_test)

# =========================
# 7. CONVERT BACK TO ORIGINAL SCALE
# =========================
# Only for temperature
temp_scaler = MinMaxScaler()
temp_scaler.fit(train_features[['meantemp']])

predictions = temp_scaler.inverse_transform(predictions.reshape(-1,1))
y_test_actual = temp_scaler.inverse_transform(y_test.reshape(-1,1))

print("Prediction done!")

# =========================
# 8. FUTURE 7 DAYS PREDICTION
# =========================
future_input = test_scaled[-seq_length:]
future_preds = []

for _ in range(7):
    future_input_reshaped = np.reshape(future_input, (1, seq_length, 3))
    pred = model.predict(future_input_reshaped)[0][0]

    # reuse last humidity & wind
    new_row = [pred, future_input[-1][1], future_input[-1][2]]
    future_preds.append(pred)

    future_input = np.vstack((future_input[1:], new_row))

# Convert back
future_preds = temp_scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

print("\nNext 7 Days Temperature Prediction:")
for i, val in enumerate(future_preds, 1):
    print(f"Day {i}: {val[0]:.2f}°C")

# =========================
# 9. PLOT RESULTS
# =========================
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual")
plt.plot(predictions, label="Predicted")

# future plot
future_x = range(len(y_test_actual), len(y_test_actual)+7)
plt.plot(future_x, future_preds, 'r--', label="Future (7 days)")

plt.legend()
plt.title("Weather Prediction using LSTM (Train + Test)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()