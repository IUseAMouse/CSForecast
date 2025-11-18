"""Data preprocessing utilities"""

import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MONTH_TRANSLATOR = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}


def reformat_trend_data(trend: List[Dict], month_translator: Dict = MONTH_TRANSLATOR) -> Dict:
    """
    Reformat trend data from HLTV format to structured format.
    
    Args:
        trend: List of trend dictionaries from HLTV
        month_translator: Dictionary mapping month names to numbers
        
    Returns:
        Dictionary with structured trend data
    """
    def remove_ordinal(date_str: str) -> str:
        """Remove ordinal suffixes from date strings."""
        return date_str.replace("th", "").replace("rd", "").replace("nd", "").replace("st", "")
    
    trend_reformat = {
        "start": [],
        "end": [],
        "maps": [],
        "rating": []
    }
    
    for period in trend:
        # Skip periods with no data
        if "displayValue" not in period:
            continue
            
        time_data = period["displayValue"].split("-")
        
        # Parse start date
        start_parts = time_data[0].split(" ")
        start_month = month_translator[start_parts[0]]
        start_day = int(remove_ordinal(start_parts[1]))
        start_year = 2000 + int(period["label"].split(" ")[1])
        
        # Parse end date
        end_parts = time_data[1].split(" ")
        end_month = month_translator[end_parts[1]]
        end_year = start_year if end_month > start_month else start_year + 1
        
        # Handle invalid day (e.g., Feb 30)
        try:
            end_day = int(remove_ordinal(end_parts[2]))
            end_date = datetime.date(end_year, end_month, end_day)
        except ValueError:
            end_day = int(remove_ordinal(end_parts[2])) - 1
            end_date = datetime.date(end_year, end_month, end_day)
        
        start_date = datetime.date(start_year, start_month, start_day)
        
        # Extract maps and rating
        maps = int(time_data[2].split(":")[-1])
        rating = float(period["value"])
        
        # Append to reformatted data
        trend_reformat["start"].append(start_date)
        trend_reformat["end"].append(end_date)
        trend_reformat["maps"].append(maps)
        trend_reformat["rating"].append(rating)
    
    return trend_reformat


def smooth_series(y: np.ndarray, window_size: int) -> np.ndarray:
    """
    Smooth a time series using a moving average.
    
    Args:
        y: Time series data
        window_size: Size of smoothing window
        
    Returns:
        Smoothed time series
    """
    box = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def preprocess_player_data(
    data: pd.DataFrame,
    smoothing_window: int = 80,
    edge_trim: int = 60
) -> pd.DataFrame:
    """
    Preprocess player data: reformat trends, smooth, and trim edges.
    
    Args:
        data: Raw player data DataFrame
        smoothing_window: Window size for smoothing
        edge_trim: Number of points to trim from edges
        
    Returns:
        Preprocessed DataFrame
    """
    # Reformat trend data
    data["rating_trend"] = data["rating_trend"].apply(
        lambda trend: reformat_trend_data(trend) if isinstance(trend, list) else trend
    )
    
    # Smooth and trim
    for trend in data["rating_trend"]:
        if not isinstance(trend, dict):
            continue
            
        rating = np.array(trend["rating"])
        rating = smooth_series(rating, smoothing_window)
        
        # Trim edges
        trend["rating"] = rating[edge_trim:-edge_trim]
        trend["start"] = trend["start"][edge_trim:-edge_trim]
        trend["end"] = trend["end"][edge_trim:-edge_trim]
        trend["maps"] = trend["maps"][edge_trim:-edge_trim]
    
    return data


def prepare_sequences(
    data: Dict,
    sequence_length: int,
    out_length: int
) -> Tuple[List, List, List, List, List]:
    """
    Prepare data sequences for training.
    
    Args:
        data: Dictionary with trend data
        sequence_length: Length of input sequences
        out_length: Length of output sequences
        
    Returns:
        Tuple of (starts, ends, maps, x_ratings, y_ratings)
    """
    starts, ends, maps_played, x_ratings, y_ratings = [], [], [], [], []
    
    max_len = len(data["rating"])
    
    for i in range(0, max_len - sequence_length - out_length - 1, sequence_length):
        starts.append(data["start"][i:i + sequence_length])
        ends.append(data["end"][i:i + sequence_length])
        maps_played.append(data["maps"][i:i + sequence_length])
        x_ratings.append(data["rating"][i:i + sequence_length])
        y_ratings.append(data["rating"][i + sequence_length:i + sequence_length + out_length])
    
    return starts, ends, maps_played, x_ratings, y_ratings