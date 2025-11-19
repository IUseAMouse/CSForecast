"""
Train forecasting models.

This script trains various models (LSTM, Transformer, GRU, Ridge, Random Forest)
on the preprocessed player data.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.csgo_forecasting.models import *
from src.csgo_forecasting.training import Trainer, prepare_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/player_data_all_cleaned.json",
        help="Path to preprocessed data (default: data/processed/player_data_all_cleaned.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "gru", "transformer", "ridge", "rf", "arima", "ets", "all"],
        help="Model to train",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        required=True,
        help="Input sequence length",
    )
    parser.add_argument(
        "--out-length",
        type=int,
        required=True,
        help="Output sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=60,
        help="Batch size (default: 60)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="Number of epochs (default: 10000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Output directory for models (default: data/models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=120,
        help="Early stopping patience (default: 120)",
    )
    
    return parser.parse_args()


def prepare_training_data(
    data: pd.DataFrame,
    seq_length: int,
    out_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training and validation data with player-based split."""
    
    train = data[data.set_split == "train"]
    
    # Split players, not sequences
    player_indices = train.index.tolist()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(player_indices)
    
    val_split = int(0.9 * len(player_indices))
    train_player_ids = set(player_indices[:val_split])
    val_player_ids = set(player_indices[val_split:])
    
    print(f"📊 Players in train: {len(train_player_ids)}")
    print(f"📊 Players in val: {len(val_player_ids)}")
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    for index, row in train.iterrows():
        trend = row["rating_trend"]
        _, _, _, x_ratings, y_ratings = prepare_data(
            trend, sequence_length=seq_length, out_length=out_length
        )
        
        # Add all sequences from this player to either train or val
        if index in train_player_ids:
            for i in range(len(y_ratings)):
                X_train.append(x_ratings[i])
                y_train.append(y_ratings[i])
        else:  # index in val_player_ids
            for i in range(len(y_ratings)):
                X_val.append(x_ratings[i])
                y_val.append(y_ratings[i])
    
    # Convert to arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Reshape for RNNs (Samples, Seq, Features)
    X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, newshape=(X_val.shape[0], X_val.shape[1], 1))
    
    return X_train, y_train, X_val, y_val


def train_lstm(
    X_train, y_train, X_val, y_val, args
):
    """Train LSTM model."""
    print("\n🧠 Training LSTM model...")
    
    model = LSTM(
        input_size=1,
        hidden_size=args.seq_length,
        output_size=args.out_length,
        num_layers=1,
        device=args.device,
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.learning_rate)
    
    # Create data loaders
    trainX = Variable(torch.Tensor(X_train)).to(args.device)
    trainY = Variable(torch.Tensor(y_train)).to(args.device)
    valX = Variable(torch.Tensor(X_val)).to(args.device)
    valY = Variable(torch.Tensor(y_val)).to(args.device)
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(trainX, trainY)
    val_dataset = TensorDataset(valX, valY)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    trainer = Trainer(
        model.model,
        criterion,
        optimizer,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        verbose=True,
        print_every=5,
    )
    
    return model, history


def train_gru(X_train, y_train, X_val, y_val, args):
    """Train GRU model."""
    print("\n🧠 Training GRU model...")
    
    model = GRU(
        input_size=1,
        hidden_size=args.seq_length,
        output_size=args.out_length,
        num_layers=1,
        device=args.device,
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.learning_rate)
    
    # Create data loaders
    trainX = Variable(torch.Tensor(X_train)).to(args.device)
    trainY = Variable(torch.Tensor(y_train)).to(args.device)
    valX = Variable(torch.Tensor(X_val)).to(args.device)
    valY = Variable(torch.Tensor(y_val)).to(args.device)
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(trainX, trainY)
    val_dataset = TensorDataset(valX, valY)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    trainer = Trainer(
        model.model,
        criterion,
        optimizer,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        verbose=True,
        print_every=5,
    )
    
    return model, history


def train_transformer(X_train, y_train, X_val, y_val, args):
    """Train Transformer model."""
    print("\n🧠 Training Transformer model...")
    
    model = TransformerModel(
        input_size=1,
        output_size=args.out_length,
        num_layers=1,
        dropout=0.2,
        nhead=4,
        d_model=64,
        device=args.device,
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.learning_rate)
    
    # Create data loaders
    trainX = Variable(torch.Tensor(X_train)).to(args.device)
    trainY = Variable(torch.Tensor(y_train)).to(args.device)
    valX = Variable(torch.Tensor(X_val)).to(args.device)
    valY = Variable(torch.Tensor(y_val)).to(args.device)
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(trainX, trainY)
    val_dataset = TensorDataset(valX, valY)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    trainer = Trainer(
        model.model,
        criterion,
        optimizer,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        verbose=True,
        print_every=5,
    )
    
    return model, history


def train_ridge(X_train, y_train, X_val, y_val, args):
    """Train Ridge regression model."""
    print("\n📊 Training Ridge Regression model...")
    
    model = RidgeModel(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Calculate validation loss
    y_pred = model.forward(X_val)
    val_loss = np.mean((y_val - y_pred) ** 2)
    
    print(f"✓ Training complete. Validation MSE: {val_loss:.5f}")
    
    return model, {"val_loss": val_loss}


def train_random_forest(X_train, y_train, X_val, y_val, args):
    """Train Random Forest model."""
    print("\n🌲 Training Random Forest model...")
    
    model = RandomForestModel(
        n_estimators=100,
        max_depth=None,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Calculate validation loss
    y_pred = model.forward(X_val)
    val_loss = np.mean((y_val - y_pred) ** 2)
    
    print(f"✓ Training complete. Validation MSE: {val_loss:.5f}")
    
    return model, {"val_loss": val_loss}


def train_arima(X_val, y_val, args):
    """
    'Train' ARIMA model.
    Actually just instantiates and runs evaluation on validation set.
    """
    print("\n📈 Configuring AutoARIMA model...")
    print("Note: ARIMA fits individual models per sample. This may take time.")
    
    model = ARIMAModel(
        out_length=args.out_length,
        max_p=5, max_q=5, max_d=2, # As per LaTeX description
        n_jobs=-1 # Use all cores
    )
    
    # We don't fit on X_train for local statistical models
    # But we evaluate on X_val to get a comparable MSE
    
    print(f"Running evaluation on {len(X_val)} validation samples...")
    
    y_pred = model.forward(X_val)
    
    # Calculate MSE
    val_loss = np.mean((y_val - y_pred) ** 2)
    
    print(f"Validation MSE: {val_loss:.5f}")
    
    return model, {"val_loss": val_loss}


def train_ets(X_val, y_val, args):
    """
    'Train' ETS model.
    """
    print("\n📉 Configuring ETS model (Auto Selection)...")
    
    model = ETSModel(
        out_length=args.out_length,
        seasonal_periods=None, # As per LaTeX description (no strict seasonality)
        n_jobs=-1
    )
    
    print(f"Running evaluation on {len(X_val)} validation samples...")
    
    y_pred = model.forward(X_val)
    val_loss = np.mean((y_val - y_pred) ** 2)
    
    print(f"Validation MSE: {val_loss:.5f}")
    
    return model, {"val_loss": val_loss}


def main():
    """Main function."""
    args = parse_args()
    
    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("\nPlease run preprocessing first:")
        print("  make preprocess")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Model Training")
    print("=" * 60)
    print(f"\n📂 Data file: {data_path}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🎯 Model: {args.model}")
    print(f"📏 Sequence length: {args.seq_length}")
    print(f"📐 Output length: {args.out_length}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🔄 Max epochs: {args.epochs}")
    print(f"📉 Learning rate: {args.learning_rate}")
    print(f"💻 Device: {args.device}")
    print("\n" + "=" * 60 + "\n")
    
    # Load data
    print("Loading data...")
    data = pd.read_json(args.data)
    print(f"✓ Loaded {len(data)} players\n")
    
    # Prepare training data
    print("Preparing training data...")
    X_train, y_train, X_val, y_val = prepare_training_data(
        data, args.seq_length, args.out_length
    )
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    
    # Models to train
    models_to_train = []
    if args.model == "all":
        models_to_train = ["lstm", "gru", "transformer", "ridge", "rf", "arima", "ets"]
    else:
        models_to_train = [args.model]
    
    # Train models
    for model_name in models_to_train:
        if model_name == "lstm":
            model, history = train_lstm(X_train, y_train, X_val, y_val, args)
        elif model_name == "gru":
            model, history = train_gru(X_train, y_train, X_val, y_val, args)
        elif model_name == "transformer":
            model, history = train_transformer(X_train, y_train, X_val, y_val, args)
        elif model_name == "ridge":
            model, history = train_ridge(X_train, y_train, X_val, y_val, args)
        elif model_name == "rf":
            model, history = train_random_forest(X_train, y_train, X_val, y_val, args)
        elif model_name == "arima":
            model, history = train_arima(X_val, y_val, args)
        elif model_name == "ets":
            model, history = train_ets(X_val, y_val, args)
        
        # Save model
        model_filename = f"{model_name}_csgo_{args.seq_length}_{args.out_length}.pickle"
        model_path = output_dir / model_filename
        model.save(str(model_path))
        print(f"💾 Model saved to: {model_path}\n")
    
    print("=" * 60)
    print("✨ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()