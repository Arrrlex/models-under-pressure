import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models_under_pressure.config import PLOTS_DIR
from models_under_pressure.interfaces.results import (
    EvaluationResult,
)


def generate_heatmap_plot(heatmap_id: str, variation_type: str, result: pd.DataFrame):
    # Create dataframe from performances

    heatmap_matrix = result.pivot(
        index="train_variation_value", columns="test_variation_value", values="accuracy"
    )

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_matrix,
        annot=True,  # Show numbers in cells
        fmt=".3f",  # Format numbers to 3 decimal places
        cmap="Blues",
        vmin=0.5,  # Minimum value for color scaling
        vmax=1.0,  # Maximum value for color scaling
        center=0.925,  # Center point for color divergence
        square=True,  # Make cells square
    )

    # Customize the plot
    plt.title(f"Probe Generalization Across {variation_type.replace('_', ' ').title()}")
    plt.xlabel("Test Variation")
    plt.ylabel("Train Variation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / f"heatmap_{heatmap_id}_{variation_type}.png")

    plt.show()


def plot_aurocs(
    result_list: list[EvaluationResult],
    title: str = "Probe Generalization to non-AIS datasets",
    bar_width: float = 0.3,
    figsize: tuple[int, int] = (8, 6),
):
    # Set width of bars and positions of the bars
    x = np.arange(len(result_list[0].dataset_name))

    plt.figure(figsize=figsize)

    # Create bars for each result
    for i, results in enumerate(result_list):
        # Extract AUROC scores from metrics
        auroc_scores = results.metrics.metrics["auroc"]
        datasets = results.dataset_name
        run_name = results.run_name

        # Create offset bars
        offset = i * bar_width
        bars = plt.bar(x + offset, auroc_scores, bar_width, label=run_name)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

    # Customize the plot
    plt.ylim(0, 1)  # AUROC ranges from 0 to 1
    plt.ylabel("AUROC Score")
    plt.title(title)

    # Center x-axis labels between grouped bars
    plt.xticks(x + bar_width / 2, datasets)
    plt.legend()

    # Save plot
    plt.savefig(PLOTS_DIR / "AUROCs.png")

    plt.show()
