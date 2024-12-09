import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)
from functions import adjust_row, remove_outliers, accuracy, smape

df = pd.read_csv("Data_from_Edo")
df = df.drop(columns=["Unnamed: 0"])

# Rimuovere gli outlier dal dataset
df = remove_outliers(df)


df_synth = df.copy()

# # Apply the function row by row 
df_synth[["Living_Area", "Price"]] = df_synth.apply(adjust_row, axis=1)
# features = ['Price',"Province", "Salary_prov_med", "Prix_m2_prov", "Municipality", "Living_Area"]
# df = df[features]
# df_synth = df_synth[features]

# Data split
train = np.array(df)[: int(0.8 * df.shape[0])]
test = np.array(df.copy())[int(0.8 * df.shape[0]) :]

synthetic_data = np.array(df_synth)[: int(0.8 * df.shape[0])]
train = np.concatenate((train, synthetic_data), axis=0)
np.random.shuffle(train)


# Normalize the features
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

X_train = train_scaled[:, 1:]
y_train = train_scaled[:, 0]
X_test = test_scaled[:, 1:]
y_test = test_scaled[:, 0]


# Function to build the model
def build_model(hp):
    model = Sequential()

    # First hidden layer
    model.add(
        Dense(
            units=hp.Int("units_layer_1", min_value=800, max_value=1000, step=50),
            activation=hp.Choice(
                "activation_layer_1", values=["relu", "tanh", "sigmoid"]
            ),
            input_dim=X_train.shape[1],
        )
    )
    # model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))

    # Second hidden layer
    if hp.Boolean("add_layer_2"):
        model.add(
            Dense(
                units=hp.Int("units_layer_2", min_value=600, max_value=800, step=50),
                activation=hp.Choice(
                    "activation_layer_2", values=["relu", "tanh", "sigmoid"]
                ),
            )
        )
    # model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))

    # Third hidden layer 
    model.add(
        Dense(
            units=hp.Int("units_layer_3", min_value=400, max_value=600, step=50),
            activation=hp.Choice(
                "activation_layer_3", values=["relu", "tanh", "sigmoid"]
            ),
        )
    )
    # model.add(Dropout(hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))

    # Fourth hidden layer 
    model.add(
        Dense(
            units=hp.Int("units_layer_4", min_value=200, max_value=400, step=50),
            activation=hp.Choice(
                "activation_layer_4", values=["relu", "tanh", "sigmoid"]
            ),
        )
    )
    # model.add(Dropout(hp.Float('dropout_4', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(
        Dense(
            units=hp.Int("units_layer_5", min_value=50, max_value=100, step=50),
            activation=hp.Choice("activation_layer_5", values=["relu", "softmax"]),
        )
    )
    # model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))

    if hp.Boolean("add_layer_6"):
        model.add(
            Dense(
                units=hp.Int("units_layer_6", min_value=20, max_value=50, step=10),
                activation=hp.Choice("activation_layer_6", values=["softmax"]),
            )
        )
        model.add(
            Dropout(hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1))
        )

    # Output layer
    model.add(Dense(1))  

    # Compilazione
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
            )
        ),
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )

    return model


tuner = kt.BayesianOptimization(
    build_model,
    objective="val_mean_squared_error",
    max_trials=30,
    directory="keras_tuner_dir",
    project_name="bayesian_tuning_example",
)

#  Hyperparameter search
tuner.search(
    X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16
)

# Best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

# # Best model
best_model = tuner.get_best_models(num_models=1)[0]

# Final training with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, mode="min", restore_best_weights=True
)
best_model.fit(
    X_train,
    y_train,
    epochs=200,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=[early_stopping],
)

# Save the model
best_model.save("model_no_outliers_16_all.h5")

# Model evaluation
y_pred_train = best_model.predict(X_train)
y_pred = best_model.predict(X_test)


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
