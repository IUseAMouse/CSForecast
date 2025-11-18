"""
Random Walk model for baseline forecasting.
"""

from typing import Optional
import numpy as np
import pickle

from .base import BaseModel


class RandomWalkModel(BaseModel):
    """
    Random Walk baseline model.
    
    Uses per-sequence adaptive estimation of step distributions.
    """

    def __init__(self, random_state: int = 42, adaptive: bool = True):
        """
        Initialize Random Walk model.

        Args:
            random_state: Random seed for reproducibility
            adaptive: If True, estimate step distribution per-sequence from input.
                     If False, use default parameters (mean=0, std=0.01)
        """
        super().__init__()
        self.random_state = random_state
        self.adaptive = adaptive
        self.output_len = 30
        self.model = self

    def build(self):
        """Build Random Walk model (just returns self)."""
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using random walk.

        Args:
            x: Input sequence of shape (batch_size, features, seq_len) or (batch_size, seq_len)

        Returns:
            Predictions of shape (batch_size, output_len)
        """
        np.random.seed(self.random_state)
        
        # Extract sequences
        if len(x.shape) == 3:
            # Shape: (batch, features, timesteps) -> extract (batch, timesteps)
            sequences = x[:, 0, :]
        elif len(x.shape) == 2:
            sequences = x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Generate predictions for each sequence
        predictions = np.zeros((len(sequences), self.output_len))
        
        for i, seq in enumerate(sequences):
            if self.adaptive:
                # Estimate from this specific sequence
                mean = np.mean(seq[1:] - seq[:-1])
                variance = np.var(seq[1:] - seq[:-1])
            else:
                # Use default parameters
                mean = 0.0
                variance = 0.01 ** 2
            
            # Handle NaN/invalid variance
            if np.isnan(variance) or variance < 1e-10:
                variance = 1e-6
            if np.isnan(mean):
                mean = 0.0
            
            # Generate random walk predictions starting from last observation
            predictions[i, 0] = seq[-1] + np.random.normal(loc=mean, scale=np.sqrt(variance))
            
            for t in range(1, self.output_len):
                predictions[i, t] = predictions[i, t-1] + np.random.normal(
                    loc=mean, scale=np.sqrt(variance)
                )
        
        return predictions

    def save(self, path: str) -> None:
        """Save model to disk."""
        state = {
            'adaptive': self.adaptive,
            'output_len': self.output_len,
            'random_state': self.random_state,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.adaptive = state.get('adaptive', True)
        self.output_len = state.get('output_len', 30)
        self.random_state = state.get('random_state', 42)