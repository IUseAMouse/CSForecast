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
    
    Uses the same logic as the existing random_walk_predictions function.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize Random Walk model.

        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.random_state = random_state
        self.step_mean = 0.0
        self.step_std = 0.01
        self.output_len = 30
        self.is_fitted = False
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
            sequences = x[:, 0, :]  # ← FIX: changed from x[:, :, -1]
        elif len(x.shape) == 2:
            sequences = x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Generate predictions for each sequence
        predictions = np.zeros((len(sequences), self.output_len))
        
        for i, seq in enumerate(sequences):
            # Use fitted parameters if available, otherwise estimate from sequence
            if self.is_fitted:
                mean = self.step_mean
                variance = self.step_std ** 2
            else:
                mean = np.mean(seq[1:] - seq[:-1])
                variance = np.var(seq[1:] - seq[:-1])
            
            # Handle NaN/invalid variance
            if np.isnan(variance) or variance < 1e-10:
                variance = 1e-6
            if np.isnan(mean):
                mean = 0.0
            
            # Generate random walk predictions
            predictions[i, 0] = seq[-1] + np.random.normal(loc=mean, scale=np.sqrt(variance))
            
            for t in range(1, self.output_len):
                predictions[i, t] = predictions[i, t-1] + np.random.normal(
                    loc=mean, scale=np.sqrt(variance)
                )
        
        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model by estimating step size distribution.

        Args:
            X: Training features
            y: Training targets
        """
        self.output_len = y.shape[1] if len(y.shape) > 1 else 1
        
        # Extract sequences
        if len(X.shape) == 3:
            # Shape: (batch, features, timesteps) -> extract (batch, timesteps)
            sequences = X[:, 0, :]  # ← FIX: changed from X[:, :, -1]
        elif len(X.shape) == 2:
            sequences = X
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
        
        # Calculate all steps
        all_steps = []
        for seq in sequences:
            # Skip sequences with NaN
            if np.any(np.isnan(seq)):
                continue
            steps = seq[1:] - seq[:-1]
            all_steps.extend(steps)
        
        # Fit distribution
        all_steps = np.array(all_steps)
        
        # Remove NaN values
        all_steps = all_steps[~np.isnan(all_steps)]
        
        if len(all_steps) == 0:
            print("⚠️  Warning: No valid steps found, using defaults")
            self.step_mean = 0.0
            self.step_std = 0.01
        else:
            self.step_mean = np.mean(all_steps)
            self.step_std = np.std(all_steps)
            
            # Ensure minimum std to avoid zero variance
            if self.step_std < 1e-6:
                self.step_std = 0.01
        
        self.is_fitted = True
        
        print(f"✓ Random Walk fitted: mean={self.step_mean:.6f}, std={self.step_std:.6f}")

    def save(self, path: str) -> None:
        """Save model to disk."""
        state = {
            'step_mean': self.step_mean,
            'step_std': self.step_std,
            'output_len': self.output_len,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.step_mean = state['step_mean']
        self.step_std = state['step_std']
        self.output_len = state['output_len']
        self.random_state = state.get('random_state', 42)
        self.is_fitted = state['is_fitted']