"""
Early stopping implementation.
"""

import numpy as np


class EarlyStopper:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 1, min_delta: float = 0):
        """
        Initialize early stopper.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            validation_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self) -> None:
        """Reset the early stopper."""
        self.counter = 0
        self.min_validation_loss = np.inf