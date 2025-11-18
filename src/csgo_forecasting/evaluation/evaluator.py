"""
Model evaluation class.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

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
        data: Any,
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

        # Vérifier le type réel du modèle
        import sklearn.base
        is_sklearn = isinstance(self.model, sklearn.base.BaseEstimator)
        
        for index, row in data.iterrows():
            trend = row["rating_trend"]
            pname = row["players_name"]
            
            _, _, _, x_ratings, y_ratings = prepare_data(
                trend, sequence_length=seq_length, out_length=out_length
            )

            if len(y_ratings) == 0:
                continue

            initial_step = x_ratings[-1]
            y_true = np.array(y_ratings[-1])

            if is_sklearn:
                # Modèle sklearn
                initial_step = np.array(initial_step)
                initial_step = np.reshape(
                    initial_step, newshape=(1, initial_step.shape[0])
                )
                y_pred = self.model.predict(initial_step)[0]
            elif is_pytorch:
                # Modèle PyTorch
                initial_step = np.array(initial_step)
                initial_step = Variable(torch.Tensor(initial_step)).to(self.device)
                initial_step = torch.reshape(
                    initial_step, shape=(1, 1, initial_step.shape[0])
                )

                with torch.no_grad():
                    predictions = self.model(initial_step)
                    
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                    
                y_pred = predictions[0].cpu().detach().numpy()
            else:
                # Classical ML model avec forward()
                initial_step = np.array(initial_step)
                initial_step = np.reshape(
                    initial_step, newshape=(1, initial_step.shape[0])
                )
                predictions = self.model.forward(initial_step)
                y_pred = np.array(predictions[0])

            metrics = calculate_metrics(y_true, y_pred)
            
            player_results.append({
                "player_name": pname,
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": y_pred,
            })

            all_predictions.append(y_pred)
            all_true.append(y_true)

            if not is_sklearn:
                torch.cuda.empty_cache()

        # Calculate aggregate metrics
        all_predictions = np.concatenate(all_predictions)
        all_true = np.concatenate(all_true)
        aggregate_metrics = calculate_metrics(all_true, all_predictions)

        return {
            "aggregate_metrics": aggregate_metrics,
            "player_results": player_results,
        }

    def plot_predictions(
        self,
        player_results: List[Dict],
        num_players: int = 15,
        figsize: tuple = (40, 40),
    ) -> None:
        """
        Plot predictions for multiple players.

        Args:
            player_results: List of player results
            num_players: Number of players to plot
            figsize: Figure size
        """
        plot_width = 5
        plot_len = min(3, (num_players + plot_width - 1) // plot_width)

        fig, ax = plt.subplots(plot_len, plot_width, figsize=figsize)
        
        for idx, result in enumerate(player_results[:num_players]):
            i = idx % plot_len
            j = idx // plot_len

            if plot_len == 1:
                curr_ax = ax[j] if plot_width > 1 else ax
            else:
                curr_ax = ax[i, j] if plot_width > 1 else ax[i]

            curr_ax.plot(result["y_true"], label="True")
            curr_ax.plot(result["y_pred"], label="Predicted")
            curr_ax.set_title(
                f"{result['player_name']}: "
                f"RMSE={result['metrics']['rmse']:.4f}"
            )
            curr_ax.legend()

        plt.tight_layout()
        plt.show()