"""
Heliogap Metrics Module.

Provides mathematical loss functions and time-series evaluation metrics
tailored for space physics and geomagnetic time-series benchmarks:
- WMAPE (Weighted Mean Absolute Percentage Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 (Coefficient of Determination)
- MDA (Mean Directional Accuracy)
- dB/dt (Rate of change of magnetic field)
"""

from typing import Union
import numpy as np
import pandas as pd


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Weighted Mean Absolute Percentage Error (WMAPE).

    WMAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100

    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted / interpolated values.

    Returns:
        float: WMAPE score as a percentage (0 to 100+).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    error_sum = np.sum(np.abs(y_true_arr - y_pred_arr))
    signal_sum = np.sum(np.abs(y_true_arr))

    if signal_sum > 0:
        return float((error_sum / signal_sum) * 100.0)
    return 0.0


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE).

    MAE = mean(|y_true - y_pred|)

    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted / interpolated values.

    Returns:
        float: MAE score.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE).

    RMSE = sqrt(mean((y_true - y_pred)^2))

    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted / interpolated values.

    Returns:
        float: RMSE score.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray, global_mean: float = None) -> float:
    """
    Calculate the Coefficient of Determination (R-squared).

    R2 = 1 - (SS_res / SS_tot)

    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted / interpolated values.
        global_mean (float, optional): Baseline mean for SS_tot calculation.
            If None, uses mean(y_true).

    Returns:
        float: R-squared score (can be negative if predictions perform worse than mean).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.size == 0:
        return 0.0

    mean_val = float(np.mean(y_true_arr)) if global_mean is None else float(global_mean)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - mean_val) ** 2)

    if ss_tot > 0:
        return float(1.0 - (ss_res / ss_tot))
    return 0.0


def calculate_mda(y_true_current: np.ndarray, y_true_previous: np.ndarray, pred_direction: Union[int, np.ndarray]) -> float:
    """
    Calculate the Mean Directional Accuracy (MDA).

    MDA measures the percentage of times the model correctly predicts the
    sign of direction (increase vs decrease) between successive steps.

    Parameters:
        y_true_current (np.ndarray): Current step ground truth values.
        y_true_previous (np.ndarray): Previous step ground truth values.
        pred_direction (int or np.ndarray): Predicted step direction signs (-1, 0, +1).

    Returns:
        float: MDA score as a percentage (0 to 100).
    """
    y_curr = np.asarray(y_true_current, dtype=np.float64)
    y_prev = np.asarray(y_true_previous, dtype=np.float64)

    if y_curr.size == 0:
        return 0.0

    true_direction = np.sign(y_curr - y_prev)
    correct_directions = np.sum(true_direction == pred_direction)
    total_points = y_curr.size

    if total_points > 0:
        return float((correct_directions / total_points) * 100.0)
    return 0.0


def calculate_dbt(data: Union[np.ndarray, pd.Series, pd.DataFrame], dt: float = 60.0, column: str = None) -> Union[np.ndarray, pd.Series]:
    """
    Calculate the time derivative of magnetic field dB/dt.

    In space physics and geomagnetism, dB/dt (time rate of change of the
    geomagnetic field) is a primary driver of geomagnetically induced
    currents (GICs).

    Parameters:
        data (np.ndarray | pd.Series | pd.DataFrame): Magnetic field data array or Series.
        dt (float): Sampling interval in seconds (default is 60.0s for 1-minute cadence).
        column (str, optional): Target column if data is a DataFrame.

    Returns:
        np.ndarray | pd.Series: Derivative values with identical length (gradient computed via central differences).
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Column name must be specified when passing a DataFrame to calculate_dbt.")
        series = data[column]
        values = series.to_numpy(dtype=np.float64)
        grad = np.gradient(values, dt)
        return pd.Series(grad, index=data.index, name=f"d{column}_dt")
    elif isinstance(data, pd.Series):
        values = data.to_numpy(dtype=np.float64)
        grad = np.gradient(values, dt)
        return pd.Series(grad, index=data.index, name=f"d{data.name or 'B'}_dt")
    else:
        values = np.asarray(data, dtype=np.float64)
        return np.gradient(values, dt)


# Concise metric aliases
wmape = calculate_wmape
mae = calculate_mae
rmse = calculate_rmse
r2 = calculate_r2
mda = calculate_mda
dbt = calculate_dbt

# Backward-compatibility aliases
calculate_dbt_simples = calculate_dbt
calcular_dbt_simples = calculate_dbt
calcular_dbt = calculate_dbt

METRIC_FUNCTIONS = {
    'wmape': calculate_wmape,
    'mae': calculate_mae,
    'rmse': calculate_rmse,
    'r2': calculate_r2,
    'mda': calculate_mda,
}