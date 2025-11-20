"""
Evaluation metrics for forecasting.
"""

from typing import Dict, List, Optional
import numpy as np
from scipy import stats
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


def calculate_horizon_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    horizons: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate metrics at specific forecast horizons.
    
    Args:
        y_true: True values of shape (n_samples, sequence_length)
        y_pred: Predicted values of shape (n_samples, sequence_length)
        horizons: List of time steps to calculate metrics at (e.g., [30, 60, 90, 120])
                 If None, uses [30, 60, 90, 120] or all available horizons
    
    Returns:
        Dictionary with horizon-specific metrics
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(1, -1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(1, -1)
    
    seq_length = y_true.shape[1]
    
    # Default horizons if not specified
    if horizons is None:
        horizons = [30, 60, 90, 120]
        # Filter to only horizons that fit in the sequence
        horizons = [h for h in horizons if h <= seq_length]
        # If sequence is shorter, use quarters of the sequence
        if not horizons:
            horizons = [seq_length // 4, seq_length // 2, 3 * seq_length // 4, seq_length]
            horizons = [h for h in horizons if h > 0]
    
    metrics = {}
    
    for horizon in horizons:
        if horizon > seq_length:
            continue
        
        # Extract predictions up to this horizon
        y_true_h = y_true[:, :horizon]
        y_pred_h = y_pred[:, :horizon]
        
        # Calculate metrics
        metrics[f"rmse@{horizon}"] = rmse(y_true_h.flatten(), y_pred_h.flatten())
        metrics[f"mae@{horizon}"] = mae(y_true_h.flatten(), y_pred_h.flatten())
        
    return metrics


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    include_horizons: bool = True,
    horizons: Optional[List[int]] = None,
    player_results: Optional[List[Dict]] = None  # NOUVEAU
) -> Dict[str, float]:
    """Calculate all metrics."""
    
    # Overall metrics
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }
    
    # Add horizon-specific metrics
    if include_horizons and player_results is not None:
        # Calculer par horizon EN AGRÉGANT LES MÉTRIQUES PAR JOUEUR
        if horizons is None:
            # Détecter out_length depuis le premier joueur
            out_length = len(player_results[0]["y_true"])
            horizons = [30, 60, 90, 120]
            horizons = [h for h in horizons if h <= out_length]
            if not horizons:
                horizons = [out_length // 4, out_length // 2, 3 * out_length // 4, out_length]
                horizons = [h for h in horizons if h > 0]
        
        # Pour chaque horizon, calculer la métrique sur chaque joueur puis moyenner
        for horizon in horizons:
            rmse_values = []
            mae_values = []
            
            for result in player_results:
                y_t = result["y_true"]
                y_p = result["y_pred"]
                
                if horizon > len(y_t):
                    continue
                    
                # Prendre les N premiers timesteps POUR CE JOUEUR
                y_t_h = y_t[:horizon]
                y_p_h = y_p[:horizon]
                
                rmse_values.append(rmse(y_t_h, y_p_h))
                mae_values.append(mae(y_t_h, y_p_h))
            
            # Moyenne sur tous les joueurs
            metrics[f"rmse@{horizon}"] = np.mean(rmse_values)
            metrics[f"mae@{horizon}"] = np.mean(mae_values)
    
    return metrics