"""
Model evaluation class.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
import pandas as pd
from scipy import stats
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sklearn.base

from .metrics import calculate_metrics
from ..training.torch import prepare_data


class Evaluator:
    """Evaluate forecasting models."""

    def __init__(self, model: Any, device: str = "cuda"):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device

    def evaluate_on_dataset(
        self,
        data: pd.DataFrame,
        seq_length: int,
        out_length: int,
        is_pytorch: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            data: Dataset to evaluate on
            seq_length: Input sequence length
            out_length: Output sequence length
            is_pytorch: Whether model is PyTorch-based

        Returns:
            Dictionary with evaluation results
        """
        all_predictions = []
        all_true = []
        player_results = []

        # Check if model is a pure sklearn estimator (not our wrapper)
        is_pure_sklearn = isinstance(self.model, sklearn.base.BaseEstimator) and not hasattr(self.model, "forward")
        
        for index, row in data.iterrows():
            trend = row["rating_trend"]
            pname = row["players_name"]
            
            # Prepare data for this specific player
            _, _, _, x_ratings, y_ratings = prepare_data(
                trend, sequence_length=seq_length, out_length=out_length
            )

            if len(y_ratings) == 0:
                continue
            
            # Take the last test set window
            initial_step = x_ratings[-1]
            y_true_sample = np.array(y_ratings[-1])

            if is_pure_sklearn:
                initial_step = np.array(initial_step)
                initial_step = np.reshape(initial_step, newshape=(1, -1))
                y_pred_sample = self.model.predict(initial_step)[0]

            elif is_pytorch:
                initial_step = np.array(initial_step)
                # Shape: (1, Seq, 1)
                initial_step = Variable(torch.Tensor(initial_step)).to(self.device)
                initial_step = torch.reshape(
                    initial_step, shape=(1, initial_step.shape[0], 1)
                )

                with torch.no_grad():
                    predictions = self.model(initial_step)
                    
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                    
                y_pred_sample = predictions[0].cpu().detach().numpy()

            else:
                initial_step = np.array(initial_step)
                # Shape: (1, Seq)
                initial_step = np.reshape(initial_step, newshape=(1, -1))
                
                try:
                    predictions = self.model.forward(initial_step, output_len=out_length)
                except TypeError:
                    # Fallback for ETS and ARIMA because they don't have an output_len arg
                    predictions = self.model.forward(initial_step)
                
                y_pred_sample = np.array(predictions[0])

            # Ensure flat arrays
            y_pred_sample = np.array(y_pred_sample).flatten()
            y_true_sample = np.array(y_true_sample).flatten()

            # Calculate metrics for this specific player/sample
            metrics = calculate_metrics(y_true_sample, y_pred_sample, include_horizons=True)

            player_results.append({
                "player_name": pname,
                "metrics": metrics,
                "y_true": y_true_sample,
                "y_pred": y_pred_sample,
            })

            all_predictions.append(y_pred_sample)
            all_true.append(y_true_sample)

            if is_pytorch:
                torch.cuda.empty_cache()

        # Calculate aggregate metrics across all players
        if not all_predictions:
            return {"aggregate_metrics": {}, "player_results": []}

        all_predictions_concat = np.concatenate(all_predictions)
        all_true_concat = np.concatenate(all_true)
        
        aggregate_metrics = calculate_metrics(
            all_true_concat, 
            all_predictions_concat, 
            include_horizons=True
        )

        return {
            "aggregate_metrics": aggregate_metrics,
            "player_results": player_results,
        }
    
    def compare_to_baseline(
        self,
        model_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        metric: str = "rmse"
    ) -> Dict[str, float]:
        """
        Perform paired t-test comparing model to baseline.
        """
        # Extract metric values per player
        model_scores = [r["metrics"][metric] for r in model_results["player_results"]]
        baseline_scores = [r["metrics"][metric] for r in baseline_results["player_results"]]
        
        # Ensure alignment
        if len(model_scores) != len(baseline_scores):
            # Fallback: intersection by player name
            model_map = {r["player_name"]: r["metrics"][metric] for r in model_results["player_results"]}
            baseline_map = {r["player_name"]: r["metrics"][metric] for r in baseline_results["player_results"]}
            
            common_names = set(model_map.keys()) & set(baseline_map.keys())
            model_scores = [model_map[n] for n in common_names]
            baseline_scores = [baseline_map[n] for n in common_names]

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(baseline_scores, model_scores)
        
        # Cohen's d
        differences = np.array(baseline_scores) - np.array(model_scores)
        std_diff = np.std(differences, ddof=1)
        cohens_d = np.mean(differences) / std_diff if std_diff != 0 else 0.0
        
        # Win rate
        wins = np.sum(np.array(model_scores) < np.array(baseline_scores))
        win_rate = 100 * wins / len(model_scores) if len(model_scores) > 0 else 0
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "very_significant": p_value < 0.001,
            "significant": p_value < 0.01,
            "95%_significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "win_rate_percent": win_rate,
            "n_players": len(model_scores)
        }

    def plot_predictions(
        self,
        player_results: List[Dict],
        num_players: int = 15,
        figsize: tuple = (20, 15), 
    ) -> None:
        """
        Plot predictions for multiple players.
        """
        if not player_results:
            print("No results to plot.")
            return

        num_players = min(num_players, len(player_results))
        cols = 3
        rows = (num_players + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        for idx in range(num_players):
            result = player_results[idx]
            ax = axes[idx]
            
            ax.plot(result["y_true"], label="True", marker='o', markersize=3)
            ax.plot(result["y_pred"], label="Predicted", linestyle='--', marker='x', markersize=3)
            ax.set_title(f"{result['player_name']}\nRMSE: {result['metrics']['rmse']:.3f}")
            if idx == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(num_players, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()