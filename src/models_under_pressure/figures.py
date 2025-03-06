import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models_under_pressure.config import PLOTS_DIR
from models_under_pressure.interfaces.results import (
    HeatmapResults,
    ProbeEvaluationResults,
)


def generate_heatmap_plot(result: HeatmapResults):
    # Create dataframe from performances
    for layer in result.layers:
        performances = result.performances[layer]
        variation_values = result.variation_values

        # Create dataframe with rows=train variations, cols=test variations
        df = pd.DataFrame(
            performances, index=variation_values, columns=variation_values
        )

        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="RdBu", vmin=0, vmax=1)

        plt.title(f"Probe Generalization Across Variations, Layer {layer}")
        plt.xlabel("Test Variation")
        plt.ylabel("Train Variation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save plot
        plt.savefig(PLOTS_DIR / f"heatmap_layer_{layer}_{result.variation_type}.png")
        plt.show()


def plot_aurocs(
    result_list: list[ProbeEvaluationResults],
    title: str = "Probe Generalization to non-AIS datasets",
    bar_width: float = 0.3,
    figsize: tuple[int, int] = (8, 6),
):
    # Set width of bars and positions of the bars
    x = np.arange(len(result_list[0].datasets))

    plt.figure(figsize=figsize)

    # Create bars for each result
    for i, results in enumerate(result_list):
        datasets = results.datasets
        auroc_scores = results.AUROC
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
