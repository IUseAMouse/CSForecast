"""
ARIMA model for time series forecasting.
Uses AutoARIMA for automatic hyperparameter selection per sample.
"""

import numpy as np
import pickle
import warnings
from typing import Optional, Tuple
from joblib import Parallel, delayed
import pmdarima as pm

from .base import BaseModel

class ARIMAModel(BaseModel):
    """
    AutoARIMA model wrapper.
    
    Unlike global models, this fits a separate ARIMA model for each input sequence
    during the forward pass, minimizing AIC.
    """

    def __init__(
        self,
        out_length: int,
        start_p: int = 0,
        max_p: int = 5,
        start_q: int = 0,
        max_q: int = 5,
        d: Optional[int] = None,
        max_d: int = 2,
        seasonal: bool = False,
        n_jobs: int = -1
    ):
        """
        Initialize AutoARIMA model.

        Args:
            out_length: Forecast horizon
            start_p: Starting value for p (AR order)
            max_p: Max value for p
            start_q: Starting value for q (MA order)
            max_q: Max value for q
            d: Order of differencing (None = let model decide)
            max_d: Max order of differencing
            seasonal: Whether to fit seasonal ARIMA (usually False for daily player data)
            n_jobs: Number of parallel jobs for batch processing (-1 = all CPUs)
        """
        super().__init__()
        self.out_length = out_length
        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.d = d
        self.max_d = max_d
        self.seasonal = seasonal
        self.n_jobs = n_jobs
        
        # No global state to learn
        self.model = None 

    def build(self):
        """No global build required for local statistical models."""
        pass

    def _fit_predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        Fit AutoARIMA on a single sequence and predict.
        
        Args:
            x: 1D array of history
            
        Returns:
            1D array of forecast
        """
        try:
            # Suppress warnings from pmdarima/statsmodels for clean output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = pm.auto_arima(
                    x,
                    start_p=self.start_p,
                    max_p=self.max_p,
                    start_q=self.start_q,
                    max_q=self.max_q,
                    d=self.d,
                    max_d=self.max_d,
                    seasonal=self.seasonal,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )
                
                forecast = model.predict(n_periods=self.out_length)
                return forecast
                
        except Exception:
            # Fallback strategy: Naive forecast (last value repeated)
            # This happens if the series is constant or too short
            return np.full(self.out_length, x[-1])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions for a batch of sequences.
        
        Args:
            x: Input array of shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            
        Returns:
            Predictions of shape (batch_size, out_length)
        """
        # Handle dimensions
        if len(x.shape) == 3:
            x = x.squeeze(-1)
        
        # Parallelize the loop over the batch
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_predict_single)(row) for row in x
        )
        
        return np.array(predictions)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Global fit is not applicable for local statistical models.
        We just store the configuration.
        """
        pass

    def save(self, path: str) -> None:
        """Save model object to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> None:
        """Load model configuration from disk."""
        with open(path, "rb") as f:
            config = pickle.load(f)
            self.__init__(**config)