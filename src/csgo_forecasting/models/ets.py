"""
ETS (Error, Trend, Seasonality) model for time series forecasting.
Implements automatic model selection based on AIC.
"""

import numpy as np
import pickle
import warnings
from typing import Optional
from joblib import Parallel, delayed
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as StatsETS

from .base import BaseModel

class ETSModel(BaseModel):
    """
    ETS model wrapper with automatic component selection.
    
    Fits multiple ETS configurations (Additive/Multiplicative trend, Damped/Undamped)
    per sample and selects the best one based on AICc.
    """

    def __init__(
        self,
        out_length: int,
        seasonal_periods: Optional[int] = None,
        n_jobs: int = -1
    ):
        """
        Initialize ETS model.

        Args:
            out_length: Forecast horizon
            seasonal_periods: Number of periods in a season (None for non-seasonal)
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        super().__init__()
        self.out_length = out_length
        self.seasonal_periods = seasonal_periods
        self.n_jobs = n_jobs
        self.model = None

    def build(self):
        pass

    def _fit_predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        Fit best ETS model on a single sequence and predict.
        """
        # Shift data to be strictly positive if we want to test Multiplicative trends
        min_val = np.min(x)
        offset = 0
        if min_val <= 0:
            offset = abs(min_val) + 1e-6
            x_proc = x + offset
        else:
            x_proc = x

        best_aic = float('inf')
        best_model_res = None
        
        configs = [
            {"error": "add", "trend": "add", "damped_trend": False},
            {"error": "add", "trend": "add", "damped_trend": True},
            {"error": "add", "trend": "mul", "damped_trend": False}, # Requires positive data
            {"error": "add", "trend": "mul", "damped_trend": True},  # Requires positive data
            {"error": "add", "trend": None, "damped_trend": False},  # Simple Exp Smoothing
        ]

        for config in configs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = StatsETS(
                        x_proc,
                        error=config["error"],
                        trend=config["trend"],
                        damped_trend=config["damped_trend"],
                        seasonal=None, # Force non-seasonal as per context
                    )
                    fit = model.fit(disp=False)
                    
                    if fit.aicc < best_aic:
                        best_aic = fit.aicc
                        best_model_res = fit
            except:
                continue
        
        if best_model_res is not None:
            forecast = best_model_res.forecast(steps=self.out_length)
            # Remove offset
            return forecast - offset
        else:
            # Fallback: Simple Exponential Smoothing (Holt) via statsmodels simple API
            # or just naive if everything fails
            return np.full(self.out_length, x[-1])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions for a batch of sequences.
        """
        if len(x.shape) == 3:
            x = x.squeeze(-1)
            
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_predict_single)(row) for row in x
        )
        
        return np.array(predictions)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No global training."""
        pass

    def save(self, path: str) -> None:
        """Save model object to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            config = pickle.load(f)
            self.__init__(**config)