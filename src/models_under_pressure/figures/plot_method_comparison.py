import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text  # <-- Import adjust_text
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from models_under_pressure.config import DATA_DIR
from models_under_pressure.experiments.monitoring_cascade import (
    get_abbreviated_model_name,
)


def plot_method_comparison(
    results_dir: Path,
    output_file: Optional[Path] = None,
    figsize: tuple[int, int] = (7, 5),
) -> None:
    results = []
    with open(results_dir / "cascade_results.jsonl") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                if result.get("fraction_of_samples") == 1.0:
                    results.append(result)

    method_results = defaultdict(list)
    for result in results:
        cascade_type = result["cascade_type"]
        model_name = result.get("baseline_model_name")

        if cascade_type == "probe":
            key = "Probe"
        elif cascade_type == "baseline":
            key = f"{get_abbreviated_model_name(model_name)} (prompted)"
            method_type = "prompted"
        elif cascade_type == "finetuned_baseline":
            key = f"{get_abbreviated_model_name(model_name)} (finetuned)"
            method_type = "finetuned"
        else:
            continue

        method_results[key].append(
            {
                "auroc": result["auroc"],
                "avg_flops_per_sample": result["avg_flops_per_sample"],
                "method_type": method_type if cascade_type != "probe" else "probe",
            }
        )

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    colors = {
        "probe": "green",
        "prompted": "blue",
        "finetuned": "red",
    }

    texts = []  # Store text annotations here

    # Create lists for legend
    legend_elements = []

    for method, data in method_results.items():
        aurocs = [d["auroc"] for d in data]
        flops = [d["avg_flops_per_sample"] for d in data]
        method_type = data[0].get("method_type", "probe")

        mean_auroc = float(np.mean(aurocs))
        mean_flops = float(np.mean(flops))

        color = colors[method_type]

        plt.plot(mean_flops, mean_auroc, "o", label=method, color=color)
        # Remove the method type suffix for display
        display_method = method.split(" (")[0] if " (" in method else method
        texts.append(
            plt.text(mean_flops, mean_auroc, display_method, ha="center", va="bottom")
        )

    # Add legend elements for method types
    for method_type, color in colors.items():
        display_name = (
            "Attention Probe"
            if method_type == "probe"
            else f"{method_type.capitalize()} LLM"
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=display_name,
            )
        )

    # Add the legend at bottom left
    plt.legend(handles=legend_elements, loc="lower left")

    plt.xlabel("FLOPs per Sample (log scale)", fontsize=12)
    plt.ylabel("Mean AUROC", fontsize=12)
    # plt.title("Method Comparison", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)

    plt.xscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"10^{int(np.log10(x))}"))

    # Adjust texts to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray"))

    plt.tight_layout()

    if output_file is None:
        output_file = results_dir / "method_comparison.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    results_dir = DATA_DIR / "results" / "monitoring_cascade_neurips"
    plot_method_comparison(results_dir)
