"""
Download player data from HLTV.org.

This script scrapes player statistics and rating trends from HLTV.org
using the metadata file containing player URLs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.csgo_forecasting.data.scraper import scrape_player_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download player data from HLTV.org"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/player_metadata.json",
        help="Path to player metadata JSON file (default: data/metadata/player_metadata.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/player_data_all.json",
        help="Output path for scraped data (default: data/raw/player_data_all.json)",
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=2,
        help="Sleep time between requests in seconds (default: 2)",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="all",
        choices=["all", "top10", "top20"],
        help="Player selection to scrape (default: all)",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if metadata file exists
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"❌ Error: Metadata file not found at {metadata_path}")
        print("\nPlease create a metadata file with the following structure:")
        print("""
{
  "data": [
    {
      "name": "player_name",
      "url": "https://www.hltv.org/stats/players/ID/player_name"
    }
  ]
}
        """)
        sys.exit(1)
    
    print("=" * 60)
    print("📥 HLTV Data Scraper")
    print("=" * 60)
    print(f"\n📂 Metadata file: {metadata_path}")
    print(f"💾 Output file: {output_path}")
    print(f"⏱️  Sleep time: {args.sleep_time}s")
    print(f"🎯 Selection: {args.selection}")
    print("\n" + "=" * 60)
    print("\n⚠️  Warning: This may take a while depending on the number of players.")
    print("Please be respectful of HLTV's servers and don't set sleep time too low.\n")
    
    # Scrape data
    try:
        data = scrape_player_data(
            metadata_path=str(metadata_path),
            output_path=str(output_path),
            sleep_time=args.sleep_time,
            selection=args.selection,
        )
        
        print("\n" + "=" * 60)
        print(f"✅ Successfully scraped data for {len(data)} players")
        print(f"💾 Data saved to: {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()