import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models_under_pressure.config import PLOTS_DIR
from models_under_pressure.interfaces.results import (
    EvaluationResult,
)

sns.set_style("darkgrid")


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
    plt.savefig(PLOTS_DIR / "AUROCs.pdf")

    plt.show()
