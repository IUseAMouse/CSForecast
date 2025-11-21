"""
Training utility functions.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable


def prepare_data(
    data: Dict[str, List],
    sequence_length: int,
    out_length: int
) -> Tuple[List, List, List, List, List]:
    """
    Prepare time series data for training.

    Args:
        data: Dictionary containing time series data
        sequence_length: Input sequence length
        out_length: Output sequence length

    Returns:
        Tuple of (starts, ends, maps, x_ratings, y_ratings)
    """
    starts = list()
    ends = list()
    maps = list()
    x_ratings = list()
    y_ratings = list()

    max_len = len(data["rating"])

    for i in range(0, max_len - sequence_length - out_length - 1, sequence_length):
        _start = data["start"][i : i + sequence_length]
        _end = data["end"][i : i + sequence_length]
        _map = data["maps"][i : i + sequence_length]
        x_rating = data["rating"][i : i + sequence_length]
        y_rating = data["rating"][
            i + sequence_length : i + sequence_length + out_length
        ]

        starts.append(_start)
        ends.append(_end)
        maps.append(_map)
        x_ratings.append(x_rating)
        y_ratings.append(y_rating)

    return starts, ends, maps, x_ratings, y_ratings


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 60,
    device: str = "cuda",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch dataloaders.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size
        device: Device to load data on

    Returns:
        Tuple of (train_loader, val_loader)
    """
    trainX = Variable(torch.Tensor(X_train)).to(device)
    trainY = Variable(torch.Tensor(y_train)).to(device)
    
    valX = Variable(torch.Tensor(X_val)).to(device)
    valY = Variable(torch.Tensor(y_val)).to(device)

    train_dataset = TensorDataset(trainX, trainY)
    val_dataset = TensorDataset(valX, valY)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def random_walk_predictions(
    initial_step: np.ndarray,
    out_length: int
) -> np.ndarray:
    """
    Generate predictions from a random walk baseline.

    Args:
        initial_step: Initial values
        out_length: Number of predictions to generate

    Returns:
        Array of predictions
    """
    initial_step = np.array(initial_step)

    mean = np.mean(initial_step[1:] - initial_step[:-1])
    variance = np.var(initial_step[1:] - initial_step[:-1])

    predictions = np.zeros(shape=out_length)
    predictions[0] = initial_step[-1] + np.random.normal(
        loc=mean, scale=np.sqrt(variance)
    )

    for i in range(1, out_length):
        predictions[i] = predictions[i - 1] + np.random.normal(
            loc=mean, scale=np.sqrt(variance)
        )

    return predictions