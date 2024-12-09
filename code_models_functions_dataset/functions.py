from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)

def remove_outliers(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Removes outliers from the DataFrame using IsolationForest.

    Parameters:
    - df (pd.DataFrame):
        The input DataFrame from which to remove outliers.
    - contamination (float):
        The proportion of outliers in the dataset. Default is 0.05 (5%).

    Returns:
    - pd.DataFrame:
        The DataFrame with outliers removed.
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(df)

    # Keep only rows predicted as inliers (label 1)
    df_filtered = df[preds == 1].reset_index(drop=True)
    return df_filtered


def adjust_row(row: pd.Series) -> pd.Series:
    """
    Adjusts the living area and price of a property with a random variation.

    This function simulates a small random change in the living area of a property
    (between -10% and +10%) and calculates the new adjusted price based on:
    1. The province's average price per square meter (weighted at 20%).
    2. The current apartment price (weighted at 80%).

    Parameters:
    - row (pd.Series):
        A row of the DataFrame containing property data.
        Expected columns:
        - 'Living_Area': The current living area in square meters.
        - 'Price': The current price of the property.
        - 'Prix_m2_prov': The average price per square meter in the province.
        - 'Price_per_m2': The price per square meter of the property.

    Returns:
    - pd.Series:
        A Series with two values:
        - 'Living_Area': The adjusted living area after the random change.
        - 'Price': The new price based on the adjusted living area.
    """
    variation = np.random.uniform(-0.10, 0.10)
    new_area = row["Living_Area"] * (1 + variation)

    price_per_m2 = row["Price"] / row["Living_Area"]

    weight_province = 0.2 * row["Prix_m2_prov"] / price_per_m2
    weight_apartment = 0.8
    adjusted_price = (
        row["Price"] * (weight_province + weight_apartment) * (1 + variation)
    )

    return pd.Series({"Living_Area": round(new_area), "Price": round(adjusted_price)})


def accuracy(y_true: np.ndarray, y_pred: np.ndarray)-> None:
    """
    Calculates and prints regression accuracy metrics including RMSE, MAE, MAPE, SMAPE, and R2.
    Returns:
    - None
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape_value = smape(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    print("RMSE:", rmse)
    print("MAE: ", mae)
    print("MAPE:", mape, "%")
    print("SMAPE:", smape_value, "%")
    print("R2:", r2)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two arrays.
    Returns:
    - float:
        The SMAPE value as a percentage.
    """
    return (
        100
        / len(y_true)
        * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    )
