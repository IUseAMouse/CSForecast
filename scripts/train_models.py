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

from src.csgo_forecasting.models import (
    LSTMModel,
    GRUModel,
    TransformerModel,
    RidgeModel,
    RandomForestModel,
)
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
        choices=["lstm", "gru", "transformer", "ridge", "random_forest", "all"],
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
        default=10000,
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
        default=50,
        help="Early stopping patience (default: 50)",
    )
    
    return parser.parse_args()


def prepare_training_data(
    data: pd.DataFrame,
    seq_length: int,
    out_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training and validation data."""
    x_all = []
    y_all = []
    
    train = data[data.set_split == "train"]
    
    for index, row in train.iterrows():
        trend = row["rating_trend"]
        _, _, _, x_ratings, y_ratings = prepare_data(
            trend, sequence_length=seq_length, out_length=out_length
        )
        
        for i in range(len(y_ratings)):
            x_all.append(x_ratings[i])
            y_all.append(y_ratings[i])
    
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    x_all = np.reshape(x_all, newshape=(x_all.shape[0], 1, x_all.shape[1]))
    
    # Split into train and validation
    train_thresh = int(0.8 * x_all.shape[0])
    
    X_train = x_all[:train_thresh]
    y_train = y_all[:train_thresh]
    X_val = x_all[train_thresh:]
    y_val = y_all[train_thresh:]
    
    return X_train, y_train, X_val, y_val


def train_lstm(
    X_train, y_train, X_val, y_val, args
):
    """Train LSTM model."""
    print("\n🧠 Training LSTM model...")
    
    model = LSTMModel(
        input_size=args.seq_length,
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
    
    model = GRUModel(
        input_size=args.seq_length,
        hidden_size=args.seq_length,
        output_size=args.out_length,
        num_layers=5,
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
        input_size=args.seq_length,
        output_size=args.out_length,
        num_layers=1,
        dropout=0.2,
        nhead=10,
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
        models_to_train = ["lstm", "gru", "transformer", "ridge", "random_forest"]
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
        elif model_name == "random_forest":
            model, history = train_random_forest(X_train, y_train, X_val, y_val, args)
        
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