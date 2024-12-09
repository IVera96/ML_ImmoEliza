import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import IsolationForest
import keras_tuner as kt
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)
from functions import adjust_row, accuracy, remove_outliers, smape


df = pd.read_csv("Data_from_Edo")
df = df.drop(columns=["Unnamed: 0"])

# features = ['Price',"Province", "Salary_prov_med", "Prix_m2_prov", "Municipality", "Living_Area"]
# df = df[features]

# Remove outliers from the dataset
df = remove_outliers(df)

# Data split
train = np.array(df)[: int(0.8 * df.shape[0])]
test = np.array(df)[int(0.8 * df.shape[0]) :]

# Normalize the features
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

X_train = train_scaled[:, 1:]
y_train = train_scaled[:, 0]
X_test = test_scaled[:, 1:]
y_test = test_scaled[:, 0]

X_train_encoded = X_train
X_test_encoded = X_test

# Function to build the model
model = Sequential()
# First hidden layer
model.add(Dense(units=1024, activation="relu", input_dim=X_train_encoded.shape[1]))
model.add(Dense(units=512, activation="relu"))
model.add(Dense(units=256, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=32, activation="softmax"))

# Output layer
model.add(Dense(1))  # Regressione, un valore continuo

# Compilation
model.compile(optimizer=Adam(learning_rate=1e-4), loss="mean_squared_error")


# Final training with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min", restore_best_weights=True
)
model.fit(
    X_train_encoded,
    y_train,
    epochs=200,
    validation_data=(X_test_encoded, y_test),
    batch_size=16,
    callbacks=[early_stopping],
)

# Save the model
model.save("simple_model.h5")

# Model evaluation
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)


def my_inverse_scaler(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Applies the inverse transformation of a scaler to a target variable, `y`, 
    while considering additional features in `X`.

    Returns:
    - np.ndarray
        The inverse scaled values of `y` as a 1D array.
    """
    tmp = y.reshape((-1, 1))
    tmp = np.concatenate((tmp, X), axis=1)
    tmp = scaler.inverse_transform(tmp)
    return tmp[:, 0]


y_pred_train = my_inverse_scaler(y_pred_train, X_train)
y_train = my_inverse_scaler(y_train, X_train)
y_test = my_inverse_scaler(y_test, X_test)
y_pred = my_inverse_scaler(y_pred, X_test)

print("Train: ")
accuracy(y_train, y_pred_train)
print("Test: ")
accuracy(y_test, y_pred)
