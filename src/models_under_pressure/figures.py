import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models_under_pressure.config import PLOTS_DIR
from models_under_pressure.interfaces.results import (
    EvaluationResult,
)

sns.set_style("darkgrid")


def generate_heatmap_plot(
    heatmap_id: str, variation_types: list[str], results: list[pd.DataFrame], mode: str
):
    label_map = {
        "Character Perspective": "Perspective",
        "Response to Situation": "Response",
        "Prompt to LLM": "Prompt",
        "Third Person": "3rd Person",
        "overly polite": "Polite",
        "angry": "Angry",
        "vulnerable": "Vulnerable",
        "casual": "Casual",
        "very short": "V. Short",
        "short": "Short",
        "medium": "Medium",
        "long": "Long",
    }
    variation_map = {
        "length": "Length",
        "language": "Language",
        "prompt_style": "Prompt Style",
        "tone": "Tone",
    }

    # Create a figure with subplots in one row
    fig, axes = plt.subplots(
        1,
        len(variation_types),
        figsize=(7 * len(variation_types) + 3, 8),
    )
    if len(variation_types) == 1:
        axes = [axes]

    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(
        left=0.1,  # Adjust if y-labels are cut off
        right=0.85,  # Make more room for colorbar
        bottom=0.15,  # Adjust if x-labels are cut off
        top=0.9,  # Adjust if titles are cut off
        wspace=0.3,  # Space between plots
    )

    for idx, (ax, variation_type, result) in enumerate(
        zip(axes, variation_types, results)
    ):
        heatmap_matrix = result.pivot(
            index="train_variation_value",
            columns="test_variation_value",
            values="accuracy",
        )

        # Apply label mapping
        heatmap_matrix.index = pd.Index(
            [label_map.get(label, label) for label in heatmap_matrix.index]
        )
        heatmap_matrix.columns = pd.Index(
            [label_map.get(label, label) for label in heatmap_matrix.columns]
        )

        # Create heatmap
        _ = sns.heatmap(
            heatmap_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.7,
            vmax=1.0,
            center=0.925,
            square=True,
            ax=ax,
            annot_kws={"size": 26 if len(heatmap_matrix.index) < 5 else 24},
            cbar=False,  # Don't show colorbar for any plot
        )

        # Customize each subplot
        if idx == 0:  # Only leftmost plot shows y-label
            ax.set_ylabel("Train Variation", fontsize=22)
        else:
            ax.set_ylabel("")
            # ax.set_yticklabels([])

        ax.set_xlabel("Test Variation", fontsize=22)
        ax.set_title(
            variation_map.get(variation_type, variation_type), fontsize=22, pad=20
        )

        # Increase tick label sizes
        ax.tick_params(axis="both", which="major", labelsize=20)
        if mode != "poster":
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

    # Add a single colorbar for the entire figure
    if mode != "poster":
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Moved slightly to the right
        fig.colorbar(axes[-1].collections[0], cax=cbar_ax, label="Accuracy")
        cbar_ax.tick_params(labelsize=20)
        cbar_ax.set_ylabel("Accuracy", fontsize=22)
    else:
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Moved slightly to the right
        _ = fig.colorbar(axes[-1].collections[0], cax=cbar_ax, ticks=[0.7, 1.0])
        cbar_ax.tick_params(labelsize=20)
        cbar_ax.set_ylabel("Accuracy", fontsize=22)

    # Save the figure
    stem = f"heatmap_{heatmap_id}_combined_{mode}"
    if mode == "poster":
        plt.savefig(PLOTS_DIR / f"{stem}.svg", bbox_inches="tight")
    else:
        plt.savefig(PLOTS_DIR / f"{stem}.pdf", bbox_inches="tight")

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
    plt.savefig(PLOTS_DIR / "AUROCs.pdf")

    plt.show()
