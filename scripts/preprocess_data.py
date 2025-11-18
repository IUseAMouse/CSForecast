"""
Preprocess raw player data.

This script loads raw player data, reformats trend data, applies smoothing,
and creates train/test splits.
"""

import argparse
import sys
from pathlib import Path
from random import shuffle

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.csgo_forecasting.data.preprocessing import preprocess_player_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw player data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw data JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save processed data",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=80,
        help="Window size for smoothing (default: 80)",
    )
    parser.add_argument(
        "--edge-trim",
        type=int,
        default=60,
        help="Number of edge points to trim (default: 60)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found at {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔧 Data Preprocessing")
    print("=" * 60)
    print(f"\n📂 Input file: {input_path}")
    print(f"💾 Output file: {output_path}")
    print(f"📊 Smoothing window: {args.smoothing_window}")
    print(f"✂️  Edge trim: {args.edge_trim}")
    print(f"📈 Test split: {args.test_split * 100}%")
    print("\n" + "=" * 60 + "\n")
    
    # Load data
    print("Loading raw data...")
    data = pd.read_json(args.input)
    print(f"✓ Loaded {len(data)} players\n")
    
    # Preprocess
    print("Preprocessing data...")
    data = preprocess_player_data(
        data,
        smoothing_window=args.smoothing_window,
        edge_trim=args.edge_trim,
    )
    print("✓ Data preprocessed\n")
    
    # Create train/test split
    print("Creating train/test split...")
    data["index"] = range(len(data))
    data = data.set_index("index")
    
    # Assign player IDs and split
    pids = list(range(len(data)))
    shuffle(pids)
    data["pid"] = pids
    
    # Calculate split index
    test_threshold = int(len(data) * (1 - args.test_split))
    
    # Assign splits
    data["set_split"] = ["train"] * len(data)
    data.loc[test_threshold:, "set_split"] = "test"
    
    train_count = len(data[data.set_split == "train"])
    test_count = len(data[data.set_split == "test"])
    
    print(f"✓ Train set: {train_count} players ({train_count/len(data)*100:.1f}%)")
    print(f"✓ Test set: {test_count} players ({test_count/len(data)*100:.1f}%)\n")
    
    # Save processed data
    print("Saving processed data...")
    data.to_json(args.output)
    print(f"✅ Data saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✨ Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()