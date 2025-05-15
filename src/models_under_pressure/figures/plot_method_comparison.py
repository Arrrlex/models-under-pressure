import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text  # <-- Import adjust_text
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ticker import FuncFormatter

from models_under_pressure.config import DATA_DIR
from models_under_pressure.experiments.monitoring_cascade import (
    get_abbreviated_model_name,
)


def darken_color(color: str, factor: float = 0.7) -> str:
    """Darken a color by the given factor (0 to 1)."""
    rgb = to_rgb(color)
    r, g, b = rgb
    darker_rgb = (r * factor, g * factor, b * factor)
    return to_hex(darker_rgb)


def plot_method_comparison(
    results_dir: Path,
    output_file: Optional[Path] = None,
    figsize: tuple[int, int] = (7, 5),
    font_size: int = 12,
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
        print(f"Processing {cascade_type} {model_name}")
        if cascade_type == "probe":
            key = "Probe"
            method_type = "probe"
        elif cascade_type == "baseline":
            key = f"{get_abbreviated_model_name(model_name)} (prompted)"
            method_type = "prompted"
        elif cascade_type == "finetuned_baseline":
            key = f"{get_abbreviated_model_name(model_name)} (finetuned)"
            method_type = "finetuned"
        else:
            print(f"Skipping {cascade_type}")
            continue

        method_results[key].append(
            {
                "auroc": result["auroc"],
                "avg_flops_per_sample": result["avg_flops_per_sample"],
                "method_type": method_type,
            }
        )

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    colors = {
        "probe": "#2563EB",
        "prompted": "#65A30D",
        "finetuned": "#C026D3",
    }

    texts = []  # Store text annotations here

    # Create lists for legend
    # legend_elements = []

    # Store points for later connection
    # probe_point = None
    # llama_8b_finetuned_point = None

    for method, data in method_results.items():
        aurocs = [d["auroc"] for d in data]
        flops = [d["avg_flops_per_sample"] for d in data]
        method_type = data[0].get("method_type", "probe")

        mean_auroc = float(np.mean(aurocs))
        mean_flops = float(np.mean(flops))

        color = colors[method_type]
        # Create darker version of the color for text
        # text_color = darken_color(color, factor=0.7)

        plt.plot(mean_flops, mean_auroc, "o", label=method, color=color, markersize=10)
        # Remove the method type suffix for display
        # display_method = method.split(" (")[0] if " (" in method else method
        # texts.append(
        #     plt.text(
        #         mean_flops,
        #         mean_auroc,
        #         display_method,
        #         ha="center",
        #         va="bottom",
        #         fontsize=font_size - 2,
        #         color=text_color,
        #         # weight="bold",
        #     )
        # )

        # Store specific points
        # if method == "Probe":
        #     probe_point = (mean_flops, mean_auroc)
        # elif "llama-8b (finetuned)" in method:
        #     llama_8b_finetuned_point = (mean_flops, mean_auroc)

    # Connect Probe and Llama-8B finetuned with a dashed line if both exist
    # if probe_point and llama_8b_finetuned_point:
    #     print(f"Connecting {probe_point} and {llama_8b_finetuned_point}")
    #     plt.plot(
    #         [probe_point[0], llama_8b_finetuned_point[0]],
    #         [probe_point[1], llama_8b_finetuned_point[1]],
    #         "--",
    #         color="gray",
    #         alpha=1.0,
    #         linewidth=2,
    #         zorder=0,  # zorder=0 to ensure it's below the points
    #     )

    #     # Add annotation about computational difference
    #     # Use logarithmic interpolation for x since it's on a log scale
    #     position_ratio = 0.4  # 0 would be at probe, 1 would be at llama-8b

    #     # Logarithmic interpolation for x-coordinate
    #     log_probe_x = np.log10(probe_point[0])
    #     log_llama_x = np.log10(llama_8b_finetuned_point[0])
    #     log_annotation_x = log_probe_x + position_ratio * (log_llama_x - log_probe_x)
    #     annotation_x = 10**log_annotation_x

    #     # Linear interpolation for y-coordinate (no log scale)
    #     annotation_y = probe_point[1] + position_ratio * (
    #         llama_8b_finetuned_point[1] - probe_point[1]
    #     )
    #     print(annotation_x, annotation_y)

    #     flops_ratio = llama_8b_finetuned_point[0] / probe_point[0]
    #     log_flops_ratio = int(np.log10(flops_ratio))

    #     # Place text directly on the line with white background
    #     plt.annotate(
    #         f"10^{log_flops_ratio} × FLOPs →",
    #         (annotation_x, annotation_y),
    #         ha="center",
    #         va="center",
    #         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    #         fontsize=font_size - 2,
    #         # weight="bold",
    #         zorder=10,  # Make sure it's on top of everything
    #     )

    # Add legend elements for method types
    # for method_type, color in colors.items():
    #     display_name = (
    #         "Attention Probe"
    #         if method_type == "probe"
    #         else f"{method_type.capitalize()} LLM"
    #     )
    #     legend_elements.append(
    #         Line2D(
    #             [0],
    #             [0],
    #             marker="o",
    #             color="w",
    #             markerfacecolor=color,
    #             markersize=9,
    #             label=display_name,
    #         )
    #     )

    # Add the legend at bottom left
    # plt.legend(handles=legend_elements, loc="lower left", fontsize=font_size - 2)

    plt.xlabel("FLOPs per Sample (log scale)", fontsize=font_size)
    plt.ylabel("Mean AUROC", fontsize=font_size)
    # plt.title("Method Comparison", fontsize=14, pad=20)
    plt.grid(True, alpha=0.6, linestyle="-", linewidth=0.7)

    # Set y-axis limit to go up to 1.0
    plt.ylim(top=1.0)

    plt.xscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"10^{int(np.log10(x))}"))

    # Adjust texts to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray"))

    # Set tick label sizes
    plt.tick_params(axis="both", which="major", labelsize=font_size - 2)

    plt.tight_layout()

    if output_file is None:
        output_file = results_dir / "method_comparison.png"
    print(f"Saving to {output_file}")
    plt.savefig(output_file, bbox_inches="tight", dpi=1000)
    plt.show()
    plt.close()


if __name__ == "__main__":
    results_dir = DATA_DIR / "results" / "monitoring_cascade_neurips"
    plot_method_comparison(results_dir, font_size=18)
