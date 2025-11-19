"""
Classical machine learning models for forecasting.
"""

from typing import Optional
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from .base import BaseModel


class RidgeModel(BaseModel):
    """Ridge regression model."""

    def __init__(self, alpha: float = 1.0):
        """
        Initialize Ridge model.

        Args:
            alpha: Regularization strength
        """
        super().__init__()
        self.alpha = alpha
        self.model = self.build()

    def build(self) -> Ridge:
        """Build Ridge model."""
        return Ridge(alpha=self.alpha)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        # Reshape if needed
        if len(x.shape) == 3:
            nsamples, nx, ny = x.shape
            x = np.reshape(x, newshape=(nsamples, nx * ny))
        elif len(x.shape) == 2 and x.shape[0] == 1:
            x = np.reshape(x, newshape=(1, x.shape[1]))
        
        return self.model.predict(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X: Training features
            y: Training targets
        """
        # Reshape if needed
        if len(X.shape) == 3:
            nsamples, nx, ny = X.shape
            X = np.reshape(X, newshape=(nsamples, nx * ny))
        
        self.model.fit(X, y)

    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)


class RandomForestModel(BaseModel):
    """Random Forest regression model."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = self.build()

    def build(self) -> RandomForestRegressor:
        """Build Random Forest model."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=1,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        # Reshape if needed
        if len(x.shape) == 3:
            nsamples, nx, ny = x.shape
            x = np.reshape(x, newshape=(nsamples, nx * ny))
        elif len(x.shape) == 2 and x.shape[0] == 1:
            x = np.reshape(x, newshape=(1, x.shape[1]))
        
        return self.model.predict(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X: Training features
            y: Training targets
        """
        # Reshape if needed
        if len(X.shape) == 3:
            nsamples, nx, ny = X.shape
            X = np.reshape(X, newshape=(nsamples, nx * ny))
        
        self.model.fit(X, y)

    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)