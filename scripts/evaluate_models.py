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

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.csgo_forecasting.evaluation import Evaluator
from src.csgo_forecasting.models import *
from src.csgo_forecasting.training.torch import prepare_data


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
        "--baseline",
        type=str,
        default="random_walk",
        help="Baseline model name for statistical comparison (default: random_walk)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.csv",
        help="Output path for results CSV (default: results/evaluation_results.csv)",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="results/statistical_tests.csv",
        help="Output path for statistical tests CSV (default: results/statistical_tests.csv)",
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
    # Force CPU loading for PyTorch models saved on CUDA
    import os
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    with open(model_path, "rb") as f:
        # Pickle will use torch.load internally, which respects map_location
        original_torch_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs['map_location'] = torch.device(device)
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_load
        try:
            model = pickle.load(f)
        finally:
            torch.load = original_torch_load
    
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


def create_baseline_model(baseline_name: str):
    """Create baseline model on-the-fly if it doesn't exist."""
    if baseline_name != "random_walk":
        return None
    
    print(f"\n🎲 Creating Random Walk baseline (adaptive)...")
    
    # No fitting needed, it's adaptive per-sequence
    model = RandomWalkModel(random_state=42, adaptive=True)
    
    print(f"  ✓ Random Walk created (will estimate per-sequence)\n")
    
    return model


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
    
    # Create output directories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output_path = Path(args.stats_output)
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📊 Model Evaluation")
    print("=" * 60)
    print(f"\n📂 Data file: {data_path}")
    print(f"🗂️  Models directory: {models_dir}")
    print(f"💾 Output file: {output_path}")
    print(f"📈 Stats output: {stats_output_path}")
    print(f"🎯 Baseline model: {args.baseline}")
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
    
    # Group models by configuration
    models_by_config = {}
    for model_file in model_files:
        model_name, seq_length, out_length = extract_config_from_filename(
            model_file.name
        )
        config_key = (seq_length, out_length)
        if config_key not in models_by_config:
            models_by_config[config_key] = []
        models_by_config[config_key].append((model_name, model_file))
    
    # Evaluate each configuration
    results = []
    statistical_results = []
    
    for (seq_length, out_length), models in models_by_config.items():
        print(f"\n{'=' * 60}")
        print(f"Configuration: seq_length={seq_length}, out_length={out_length}")
        print(f"{'=' * 60}\n")
        
        # Find baseline model for this configuration
        baseline_file = None
        baseline_name = None
        for model_name, model_file in models:
            if model_name == args.baseline:
                baseline_file = model_file
                baseline_name = model_name
                break
        
        baseline_results = None
        baseline_model = None

        # Try to load baseline from file
        if baseline_file is not None:
            print(f"📊 Evaluating baseline: {baseline_name}")
            baseline_model = load_model(baseline_file, args.device)
        else:
            # Create baseline on-the-fly if it's random_walk
            print(f"⚠️  No baseline file found for {args.baseline}")
            baseline_model = create_baseline_model(args.baseline)
            baseline_name = args.baseline

        # Evaluate baseline if available
        if baseline_model is not None:
            print(f"📊 Evaluating baseline: {args.baseline}")
            baseline_evaluator = Evaluator(baseline_model, device=args.device)
            baseline_results = baseline_evaluator.evaluate_on_dataset(
                test_data,
                seq_length=seq_length,
                out_length=out_length,
                is_pytorch=False,
            )
            baseline_metrics = baseline_results["aggregate_metrics"]
            print(f"✓ Baseline RMSE: {baseline_metrics['rmse']:.6f}\n")
            
            # Store baseline results
            results.append({
                "model": args.baseline,
                "seq_length": seq_length,
                "out_length": out_length,
                **baseline_metrics,
            })
        else:
            print(f"⚠️  Could not create baseline model\n")
        
        # Evaluate other models
        for model_name, model_file in models:
            if model_name == args.baseline:
                continue  # Skip baseline, already evaluated
            
            print(f"\n{'-' * 60}")
            print(f"Evaluating: {model_name}")
            print(f"{'-' * 60}\n")
            
            # Load model
            print("Loading model...")
            model = load_model(model_file, args.device)
            print(f"✓ Model loaded\n")
            
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
            print("\n📈 Overall Metrics:")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE:  {metrics['mae']:.6f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  R²:   {metrics['r2']:.6f}")
            print(f"  MSE:  {metrics['mse']:.6f}")

            # Print horizon-specific metrics
            horizon_keys = [k for k in metrics.keys() if '@' in k]
            if horizon_keys:
                print("\n📈 Horizon-Specific Metrics:")
                for key in sorted(horizon_keys):
                    value = metrics[key]
                    print(f"  {key.upper()}: {value:.6f}")
            
            # Store results
            results.append({
                "model": model_name,
                "seq_length": seq_length,
                "out_length": out_length,
                **metrics,
            })
            
            # Statistical comparison with baseline
            if baseline_results is not None:
                print("\n🧪 Statistical comparison with baseline...")
                stats_comparison = evaluator.compare_to_baseline(
                    eval_results,
                    baseline_results,
                    metric="rmse"
                )
                
                improvement = (
                    (baseline_metrics['rmse'] - metrics['rmse']) 
                    / baseline_metrics['rmse'] * 100
                )
                
                print(f"  Improvement: {improvement:+.2f}%")
                print(f"  p-value: {stats_comparison['p_value']:.6f}")
                print(f"  Significant (p<0.001): {'✓ Yes' if stats_comparison['very_significant'] else '✗ No'}")
                print(f"  Significant (p<0.01): {'✓ Yes' if stats_comparison['significant'] else '✗ No'}")
                print(f"  Significant (p<0.05): {'✓ Yes' if stats_comparison['95%_significant'] else '✗ No'}")
                print(f"  Win rate: {stats_comparison['win_rate_percent']:.1f}%")
                print(f"  Cohen's d: {stats_comparison['cohens_d']:.3f}")
                
                # Store statistical results
                statistical_results.append({
                    "model": model_name,
                    "baseline": baseline_name,
                    "seq_length": seq_length,
                    "out_length": out_length,
                    "improvement_percent": improvement,
                    "t_statistic": stats_comparison["t_statistic"],
                    "p_value": stats_comparison["p_value"],
                    "significant_at_001": stats_comparison["very_significant"],
                    "significant_at_01": stats_comparison["significant"],
                    "significant_at_05": stats_comparison["95%_significant"],
                    "win_rate_percent": stats_comparison["win_rate_percent"],
                    "cohens_d": stats_comparison["cohens_d"],
                    "n_players": stats_comparison["n_players"],
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
    
    # Save statistical results
    if statistical_results:
        stats_df = pd.DataFrame(statistical_results)
        stats_df = stats_df.sort_values(["out_length", "p_value"])
        stats_df.to_csv(stats_output_path, index=False)
        print(f"✓ Statistical tests saved to: {stats_output_path}")
    
    # Print summary tables
    print(f"\n{'=' * 60}")
    print("📊 Performance Summary")
    print(f"{'=' * 60}\n")
    print(results_df.to_string(index=False))
    
    if statistical_results:
        print(f"\n{'=' * 60}")
        print("🧪 Statistical Tests Summary")
        print(f"{'=' * 60}\n")
        print(stats_df[['model', 'out_length', 'improvement_percent', 
                       'p_value', 'significant_at_001', 'significant_at_01',
                       'significant_at_05', 'win_rate_percent', 
                       'cohens_d']].to_string(index=False))
    
    print(f"\n{'=' * 60}")
    print("✨ Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()