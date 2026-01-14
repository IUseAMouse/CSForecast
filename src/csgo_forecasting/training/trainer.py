"""
Generic trainer class for all models.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from tqdm import tqdm

from .early_stopping import EarlyStopper


class Trainer:
    """Generic trainer for forecasting models."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        early_stopping_patience: int = 50,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            early_stopping_patience: Patience for early stopping
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.early_stopper = EarlyStopper(patience=early_stopping_patience)
        
        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_x)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = self.criterion(outputs, batch_y)
            epoch_loss += loss.item()
            num_batches += 1

            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()

        return epoch_loss / num_batches

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
                num_batches += 1

                torch.cuda.empty_cache()

        return val_loss / num_batches

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 1000,
        verbose: bool = True,
        print_every: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            verbose: Whether to print progress
            print_every: Print frequency

        Returns:
            Dictionary with training history
        """
        best_val_loss = float("inf")
        
        iterator = tqdm(range(num_epochs)) if verbose else range(num_epochs)
        
        for epoch in iterator:
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if self.early_stopper.early_stop(val_loss):
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

            if verbose and epoch % print_every == 0:
                print(
                    f"Epoch: {epoch}, "
                    f"Train Loss: {train_loss:.5f}, "
                    f"Val Loss: {val_loss:.5f}"
                )

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }