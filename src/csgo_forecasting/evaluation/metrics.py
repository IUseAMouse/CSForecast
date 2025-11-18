"""
Evaluation metrics for forecasting.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² score.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² value
    """
    return r2_score(y_true, y_pred)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with all metrics
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }