import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from models_under_pressure.config import DATA_DIR
from models_under_pressure.experiments.monitoring_cascade import (
    get_abbreviated_model_name,
)


def plot_method_comparison(
    results_dir: Path,
    output_file: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot comparison of different methods showing the tradeoff between FLOPs and AUROC.

    Args:
        results_dir: Directory containing the results files
        output_file: Path to save the plot. If None, saves to results_dir / "method_comparison.pdf"
        figsize: Figure size as (width, height) in inches
    """
    # Read results from file
    results = []
    with open(results_dir / "cascade_results.jsonl") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                # Only include results with fraction_of_samples = 1.0
                if result.get("fraction_of_samples") == 1.0:
                    results.append(result)

    # Group results by method type and model
    method_results = defaultdict(list)
    for result in results:
        cascade_type = result["cascade_type"]
        model_name = result.get("baseline_model_name")

        if cascade_type == "probe":
            key = "Probe"
        elif cascade_type == "baseline":
            key = f"{get_abbreviated_model_name(model_name)} (pr.)"
        elif cascade_type == "finetuned_baseline":
            key = f"{get_abbreviated_model_name(model_name)} (ft.)"
        else:
            continue  # Skip other cascade types

        method_results[key].append(
            {
                "auroc": result["auroc"],
                "avg_flops_per_sample": result["avg_flops_per_sample"],
            }
        )

    # Set up the plot
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    # Plot each method
    for method, data in method_results.items():
        # Calculate mean for this method
        aurocs = [d["auroc"] for d in data]
        flops = [d["avg_flops_per_sample"] for d in data]

        mean_auroc = float(np.mean(aurocs))
        mean_flops = float(np.mean(flops))

        # Plot point
        plt.plot(
            mean_flops,
            mean_auroc,
            "o",
            label=method,
        )

        # Add method name above point
        plt.annotate(
            method,
            (mean_flops, mean_auroc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Customize plot
    plt.xlabel("Average FLOPs per Sample (log scale)", fontsize=12)
    plt.ylabel("AUROC", fontsize=12)
    plt.title("Method Comparison", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)

    # Set x-axis to log scale
    plt.xscale("log")

    # Format x-axis ticks to be more readable
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"10^{int(np.log10(x))}"))

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if output_file is None:
        output_file = results_dir / "method_comparison.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Example usage
    results_dir = DATA_DIR / "results" / "monitoring_cascade_neurips"
    plot_method_comparison(results_dir)
