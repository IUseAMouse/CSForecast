"""
Evaluate trained models.

This script loads trained models and evaluates them on the test set,
computing various metrics and optionally generating visualizations.
"""

import argparse
import sys
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.csgo_forecasting.evaluation import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/player_data_all_cleaned.json",
        help="Path to preprocessed data (default: data/processed/player_data_all_cleaned.json)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.csv",
        help="Output path for results CSV (default: results/evaluation_results.csv)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate prediction plots",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=15,
        help="Number of players to plot (default: 15)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available)",
    )
    
    return parser.parse_args()


def load_model(model_path: Path, device: str):
    """Load a trained model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    if hasattr(model, "to"):
        model = model.to(device)
    
    return model


def extract_config_from_filename(filename: str):
    """Extract model name and configuration from filename."""
    # Format: modelname_csgo_seqlength_outlength.pickle
    parts = filename.replace(".pickle", "").split("_")
    
    model_name = parts[0]
    seq_length = int(parts[2])
    out_length = int(parts[3])
    
    return model_name, seq_length, out_length


def main():
    """Main function."""
    args = parse_args()
    
    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Check if models directory exists
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Error: Models directory not found at {models_dir}")
        print("\nPlease train models first:")
        print("  make train-all")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📊 Model Evaluation")
    print("=" * 60)
    print(f"\n📂 Data file: {data_path}")
    print(f"🗂️  Models directory: {models_dir}")
    print(f"💾 Output file: {output_path}")
    print(f"💻 Device: {args.device}")
    print("\n" + "=" * 60 + "\n")
    
    # Load data
    print("Loading data...")
    data = pd.read_json(args.data)
    test_data = data[data.set_split == "test"]
    print(f"✓ Loaded {len(test_data)} test players\n")
    
    # Find all model files
    model_files = list(models_dir.glob("*.pickle"))
    
    if not model_files:
        print(f"❌ No model files found in {models_dir}")
        sys.exit(1)
    
    print(f"Found {len(model_files)} model(s) to evaluate\n")
    
    # Evaluate each model
    results = []
    
    for model_file in model_files:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_file.name}")
        print(f"{'=' * 60}\n")
        
        # Extract configuration
        model_name, seq_length, out_length = extract_config_from_filename(
            model_file.name
        )
        
        # Load model
        print("Loading model...")
        model = load_model(model_file, args.device)
        print(f"✓ Model loaded: {model_name}\n")
        
        # Determine if PyTorch model
        is_pytorch = model_name in ["lstm", "gru", "transformer"]
        
        # Evaluate
        print("Evaluating on test set...")
        evaluator = Evaluator(model, device=args.device)
        
        eval_results = evaluator.evaluate_on_dataset(
            test_data,
            seq_length=seq_length,
            out_length=out_length,
            is_pytorch=is_pytorch,
        )
        
        # Print metrics
        metrics = eval_results["aggregate_metrics"]
        print("\n📈 Metrics:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  R²:   {metrics['r2']:.6f}")
        print(f"  MSE:  {metrics['mse']:.6f}")
        
        # Store results
        results.append({
            "model": model_name,
            "seq_length": seq_length,
            "out_length": out_length,
            **metrics,
        })
        
        # Generate plots if requested
        if args.plot:
            print(f"\nGenerating plots for {args.num_plots} players...")
            evaluator.plot_predictions(
                eval_results["player_results"],
                num_players=args.num_plots,
            )
    
    # Save results to CSV
    print(f"\n{'=' * 60}")
    print("Saving results...")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["out_length", "rmse"])
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved to: {output_path}")
    
    # Print summary table
    print(f"\n{'=' * 60}")
    print("📊 Summary")
    print(f"{'=' * 60}\n")
    print(results_df.to_string(index=False))
    
    print(f"\n{'=' * 60}")
    print("✨ Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()