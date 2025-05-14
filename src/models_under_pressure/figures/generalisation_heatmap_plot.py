from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models_under_pressure.config import PLOTS_DIR, PROJECT_ROOT
from models_under_pressure.interfaces.results import HeatmapRunResults
from models_under_pressure.utils import generate_short_id


def rename_variation_value(variation_value: str) -> str:
    return {
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
    }.get(variation_value, variation_value)


def rename_variation_type(variation_type: str) -> str:
    return {
        "length": "Length",
        "language": "Language",
        "prompt_style": "Prompt Style",
        "tone": "Tone",
    }.get(variation_type, variation_type)


def rename_metric(metric: str) -> str:
    return {
        "accuracy": "Accuracy",
        "tpr_at_1pct_fpr": "TPR @ 1% FPR",
        "auroc": "AUROC",
    }.get(metric, metric)


def generate_heatmap_plot(
    heatmaps: dict[str, pd.DataFrame],
    metric: str,
    plot_id: str,
) -> None:
    # Create a figure with subplots in one row
    fig, axes = plt.subplots(
        1,
        len(heatmaps),
        figsize=(7 * len(heatmaps) + 3, 8),
    )
    if len(heatmaps) == 1:
        axes = [axes]

    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(
        left=0.1,  # Adjust if y-labels are cut off
        right=0.85,  # Make more room for colorbar
        bottom=0.15,  # Adjust if x-labels are cut off
        top=0.9,  # Adjust if titles are cut off
        wspace=0.3,  # Space between plots
    )

    for idx, (ax, (variation_type, heatmap)) in enumerate(zip(axes, heatmaps.items())):
        heatmap_matrix = heatmap.pivot(
            index="train_variation_value",
            columns="test_variation_value",
            values=metric,
        )

        # Apply label mapping
        heatmap_matrix.index = pd.Index(
            [rename_variation_value(label) for label in heatmap_matrix.index]
        )
        heatmap_matrix.columns = pd.Index(
            [rename_variation_value(label) for label in heatmap_matrix.columns]
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
        ax.set_title(rename_variation_type(variation_type), fontsize=22, pad=20)

        # Increase tick label sizes
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

        # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Moved slightly to the right
    fig.colorbar(axes[-1].collections[0], cax=cbar_ax, label=rename_metric(metric))
    cbar_ax.tick_params(labelsize=20)
    cbar_ax.set_ylabel(rename_metric(metric), fontsize=22)

    # Save the figure
    plt.savefig(
        PLOTS_DIR / f"heatmap_{plot_id}_{metric}.svg",
        bbox_inches="tight",
    )
    plt.savefig(
        PLOTS_DIR / f"heatmap_{plot_id}_{metric}.pdf",
        bbox_inches="tight",
    )

    plt.show()


def main(*plot_paths: Path):
    results = [
        HeatmapRunResults.model_validate_json(line)
        for plot_path in plot_paths
        for line in open(plot_path).readlines()
    ]

    heatmaps = [result.heatmaps() for result in results]

    # Check for overlapping variation types between results)
    all_keys = [set(r.heatmaps().keys()) for r in results]
    if any(k1 & k2 for i, k1 in enumerate(all_keys) for k2 in all_keys[i + 1 :]):
        raise ValueError("Results contain overlapping variation types")

    generate_heatmap_plot(
        heatmaps={k: v for heatmap in heatmaps for k, v in heatmap.items()},
        metric="accuracy",
        plot_id=generate_short_id(),
    )


if __name__ == "__main__":
    heatmap_dir = PROJECT_ROOT / "data/results/generate_heatmaps"
    plot_paths = [
        heatmap_dir / "results_hUjuvHef.jsonl",
        heatmap_dir / "results_II9z56Bi.jsonl",
        heatmap_dir / "results_MWa6YOfW.jsonl",
        heatmap_dir / "results_s0L9PnXM.jsonl",
    ]
    main(*plot_paths)
