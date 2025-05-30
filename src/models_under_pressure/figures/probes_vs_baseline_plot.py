import colorsys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import RESULTS_DIR
from models_under_pressure.figures.utils import (
    get_baseline_results,
    get_continuation_results,
    get_probe_results,
)


def calculate_cascade_and_auroc(group: pd.DataFrame) -> float:
    group["probe_baseline_and_score"] = group["probe_labels"] * group["baseline_labels"]

    return float(
        roc_auc_score(
            group["probe_ground_truth_labels"], group["probe_baseline_and_score"]
        )
    )


def prepare_data(
    probe_paths: list[Path],
    baseline_paths: list[Path],
    continuation_paths: list[Path],
) -> pd.DataFrame:
    probe_results = get_probe_results(probe_paths)
    baseline_results = get_baseline_results(baseline_paths)
    continuation_results = get_continuation_results(continuation_paths)

    """
    Load the probe and baseline results, combine the dataframes on the ids and dataset_name
    ensuring the probe and baseline results have different column names

    """

    # Rename columns output_scores, and output_labels to scores and labels
    probe_results.rename(
        columns={"output_scores": "scores", "output_labels": "labels"},
        inplace=True,
    )
    # The labels here are ground truth...
    baseline_results.rename(
        columns={
            "scores": "scores",
            "ground_truth": "ground_truth_labels",
        },
        inplace=True,
    )
    baseline_results["labels"] = (baseline_results["scores"] > 0.5).astype(int)

    # The labels here aren't ground truth...
    continuation_results.rename(
        columns={
            "high_stakes_scores": "scores",
            "ground_truth": "ground_truth_labels",
        },
        inplace=True,
    )
    continuation_results["labels"] = (continuation_results["scores"] > 0.5).astype(int)

    continuation_results["method"] = "continuation_" + continuation_results[
        "model_name"
    ].apply(lambda x: x.split("/")[-1])
    baseline_results["method"] = "baseline_" + baseline_results["model_name"].apply(
        lambda x: x.split("/")[-1]
    )
    probe_results["method"] = "probe_" + probe_results["probe_name"].apply(
        lambda x: x.split("/")[-1]
    )

    combined_df_columns = [
        "dataset_name",
        "ids",
        "method",
        "scores",
        "labels",
        "ground_truth_labels",
        "load_id",
    ]

    # For each probe, baseline and continuation dataset
    df_combined_results = pd.concat(
        [
            probe_results[combined_df_columns],
            baseline_results[combined_df_columns],
            continuation_results[combined_df_columns],
        ],
        ignore_index=True,
    )

    return df_combined_results


def create_plot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the dataframe that will eventually be used to create the plot.

    """

    # Calculate the AUROC of a aggregated pandas group
    def calc_auroc(group: pd.DataFrame) -> float:
        return float(roc_auc_score(group["ground_truth_labels"], group["scores"]))

    # Calculate TPR at 1% FPR for a group
    def calc_tpr_at_fpr(group: pd.DataFrame, fpr_target: float = 0.01) -> float:
        """Calculate TPR at a fixed FPR threshold for a group."""
        from models_under_pressure.probes.metrics import tpr_at_fixed_fpr_score

        return tpr_at_fixed_fpr_score(
            group["ground_truth_labels"].values, group["scores"].values, fpr=fpr_target
        )

    # Group by dataset, method, and load_id and calculate AUROC
    auroc_grp = (
        df.groupby(["dataset_name", "method", "load_id"])
        .apply(lambda x: float(roc_auc_score(x["ground_truth_labels"], x["scores"])))
        .reset_index()
        .rename(columns={0: "auroc"})
    )

    # Group by dataset, method, and load_id and calculate TPR at 1% FPR
    tpr_grp = (
        df.groupby(["dataset_name", "method", "load_id"])
        .apply(lambda x: calc_tpr_at_fpr(x, fpr_target=0.01))
        .reset_index()
        .rename(columns={0: "tpr_at_fpr"})
    )

    # Merge the two dataframes
    grp = pd.merge(auroc_grp, tpr_grp, on=["dataset_name", "method", "load_id"])

    # Now group by the dataset_name and method and calculate the mean auroc and std of the auroc column:
    plot_df = grp.groupby(["dataset_name", "method"]).agg(
        auroc=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        auroc_err=(
            "auroc",
            lambda x: 1.96 * x.std() / np.sqrt(len(x)),
        ),  # 95% confidence interval
        tpr_at_fpr=("tpr_at_fpr", "mean"),
        tpr_at_fpr_std=("tpr_at_fpr", "std"),
        tpr_at_fpr_err=(
            "tpr_at_fpr",
            lambda x: 1.96 * x.std() / np.sqrt(len(x)),
        ),  # 95% confidence interval
    )

    return plot_df.reset_index()


def matplotlib_to_hsv(color_name: str) -> dict:
    """Convert a matplotlib color to HSV values with human-readable formatting."""
    # Convert to RGB first (values in range 0–1)
    rgb = mcolors.to_rgb(color_name)

    # Convert RGB to HSV
    hsv = colorsys.rgb_to_hsv(*rgb)  # Returns (h, s, v) with h in 0–1, s and v in 0–1

    # Convert to degrees and percentages (for use in color pickers)
    h_deg = round(hsv[0] * 360)
    s_pct = round(hsv[1] * 100)
    v_pct = round(hsv[2] * 100)

    return {
        "hue_degrees": h_deg,
        "saturation_percent": s_pct,
        "brightness_percent": v_pct,
        "rgb": rgb,
        "hex": mcolors.to_hex(rgb),
    }


def print_color_info(name: str, color: str) -> None:
    """Print color information in a readable format."""
    hsv = matplotlib_to_hsv(color)
    print(f"Color: {name}")
    print(
        f"  HSV: {hsv['hue_degrees']}°, {hsv['saturation_percent']}%, {hsv['brightness_percent']}%"
    )
    print(f"  HEX: {hsv['hex']}")
    print(f"  RGB: {hsv['rgb']}")
    print()


def hsv_to_rgb(h, s, v):
    """Convert HSV color values to RGB hex string for matplotlib"""

    # Convert HSV to RGB (values between 0 and 1)
    r, g, b = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)

    # Convert to hex string format
    return mcolors.rgb2hex((r, g, b))


def plot_results(
    plot_df: pd.DataFrame, metric: str = "auroc", show_legend: bool = False
) -> None:
    """
    Plot the results as a grouped bar chart with error bars where available.
    Probe methods are plotted first with distinct colors.
    Finetuned and continuation methods share colors but have different patterns.

    Args:
        plot_df: DataFrame containing the results to plot
        metric: Which metric to use for plotting. One of "auroc" or "tpr_at_fpr"
        show_legend: Whether to display the legend
    """

    if metric not in ["auroc", "tpr_at_fpr"]:
        raise ValueError(f"Invalid metric: {metric}. Must be 'auroc' or 'tpr_at_fpr'")

    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))

    # Define utility functions for model name and size extraction
    def extract_model_name(label: str) -> str:
        """Extract the base model name (e.g., llama, gemma) from the label."""
        import re

        # Look for common model names - adapt this regex as needed
        name_match = re.search(r"(?i)(llama|gemma|mistral|gpt|opt|phi)", label)
        if name_match:
            return name_match.group(1).lower()  # Convert to lowercase for consistency
        return ""  # Default if no model name found

    def extract_model_size(label: str) -> int:
        """Extract the model size (in billions) from the label."""
        import re

        size_match = re.search(r"(\d+)[bB]", label)
        if size_match:
            return int(size_match.group(1))
        return 0  # Default if no size found

    # Get unique datasets and methods
    datasets = plot_df["dataset_name"].unique()

    # Filter datasets based on metric
    if metric == "tpr_at_fpr":
        # Remove MTS dataset for TPR@FPR metric
        datasets = np.array([d for d in datasets if "MTS" not in d])

    # Sort alphabetically for consistency
    datasets = np.array(sorted(datasets))

    all_methods = plot_df["method"].unique()

    # Separate methods by type
    probe_methods = sorted([m for m in all_methods if m.startswith("probe_")])

    # Include all finetuned baselines (both Gemma and Llama)
    finetuned_methods = sorted([m for m in all_methods if m.startswith("baseline_")])

    # Include all continuation baselines (both Gemma and Llama)
    continuation_methods = sorted(
        [m for m in all_methods if m.startswith("continuation_")]
    )

    # Custom sort function to order models by name then size
    def sort_by_model_name_and_size(method_list: list[str]) -> list[str]:
        return sorted(
            method_list,
            key=lambda x: (
                extract_model_name(x.split("_")[1]),
                extract_model_size(x.split("_")[1]),
            ),
        )

    # Sort methods by model name and size to match legend ordering
    probe_methods = sort_by_model_name_and_size(probe_methods)
    finetuned_methods = sort_by_model_name_and_size(finetuned_methods)
    continuation_methods = sort_by_model_name_and_size(continuation_methods)

    # Order methods with probes first, then finetuned, then continuations
    methods = probe_methods + finetuned_methods + continuation_methods

    # Define distinct colors for different method types
    probe_colors = {
        "probe_attention": "#ff7f0e",  # Orange for attention probe
        "probe_softmax": "#1f77b4",  # Blue for softmax probe
    }
    finetuned_colors = [
        "#7CFC00",
        "#7CFC00",
        "#006400",
        "#006400",
    ]
    continuation_colors = [
        "#9999FF",
        "#9999FF",
        (0.0, 0.5019607843137255, 0.5019607843137255),
        (0.0, 0.5019607843137255, 0.5019607843137255),
        (0.0, 0.5019607843137255, 0.5019607843137255),
    ]

    # Print HSV values for all colors
    print("\n=== COLOR INFORMATION ===")
    print("PROBE COLORS:")
    for name, color in probe_colors.items():
        print_color_info(name, color)

    print("FINETUNED COLORS:")
    for i, color in enumerate(finetuned_colors):
        print_color_info(f"finetuned_{i}", color)

    print("CONTINUATION COLORS:")
    for i, color in enumerate(continuation_colors):
        print_color_info(f"continuation_{i}", color)
    print("========================\n")

    # Create a color dictionary
    color_dict = {}

    # Assign colors to probe methods based on whether they contain "attention" or "softmax"
    for method in probe_methods:
        if "attention" in method:
            color_dict[method] = probe_colors["probe_attention"]
        elif "softmax" in method:
            color_dict[method] = probe_colors["probe_softmax"]
        else:
            # Fallback for any other probe types
            color_dict[method] = "#2ca02c"  # Default green for unrecognized probes

    # Assign colors to finetuned methods
    for i, method in enumerate(finetuned_methods):
        color_dict[method] = finetuned_colors[i % len(finetuned_colors)]

    # Assign colors to continuation methods
    for i, method in enumerate(continuation_methods):
        color_dict[method] = continuation_colors[i % len(continuation_colors)]

    # Define hatch patterns with variations for different methods and sizes
    hatch_patterns = {
        # Small models (1b, 3b) - small circles
        "small_1": ".",  # Tiny dots for 1b
        "small_3": "o",  # Small circles for 3b
        # Medium models (8b, 12b) - medium circles
        "medium_8": "o",  # Medium circles for 8b
        "medium_12": "o",  # Larger circles for 12b
        # Large models (>12b) - large circles
        "large_27": "O",  # Large circles for 27b
        "large_70": "O",  # Largest circles for 70b
    }

    # Store linewidth information for different model sizes
    linewidth_dict = {}
    # Store hatchdensity (spacing between elements) for different model sizes
    # Higher value means sparser hatching (fewer circles)
    hatchdensity_dict = {}

    # Define hatch patterns based on model size
    hatch_dict = {}
    # No hatching for probe methods
    for method in probe_methods:
        hatch_dict[method] = ""
        linewidth_dict[method] = 0.5
        hatchdensity_dict[method] = 1

    # Assign hatches based on exact model size to control thickness
    for method in finetuned_methods + continuation_methods:
        # Extract model name part after the prefix
        if method.startswith("baseline_"):
            model_name = method.split("_")[1]
        else:  # continuation
            model_name = method.split("_")[1]

        # Get model size
        size = extract_model_size(model_name)

        # Assign pattern based on exact size for specific thickness
        if size == 1:
            hatch_dict[method] = hatch_patterns["small_1"]
            linewidth_dict[method] = 0.8
            hatchdensity_dict[method] = 4  # More dense (many small dots)
        elif size <= 3:
            hatch_dict[method] = hatch_patterns["small_3"]
            linewidth_dict[method] = 1.0
            hatchdensity_dict[method] = 6  # Dense small circles
        elif size <= 8:
            hatch_dict[method] = hatch_patterns["medium_8"]
            linewidth_dict[method] = 1.2
            hatchdensity_dict[method] = 8  # Medium spacing
        elif size <= 12 or "gemma-12b" in model_name.lower():
            hatch_dict[method] = hatch_patterns["medium_12"]
            linewidth_dict[method] = 1.5
            hatchdensity_dict[method] = 10  # Fewer medium circles
        elif size <= 27:
            hatch_dict[method] = hatch_patterns["large_27"]
            linewidth_dict[method] = 1.8
            hatchdensity_dict[method] = 12  # Few large circles
        else:  # > 27b (like 70b)
            hatch_dict[method] = hatch_patterns["large_70"]
            linewidth_dict[method] = 2.0
            hatchdensity_dict[method] = 15  # Very few very large circles

    # Set up the bar positions
    bar_width = 0.8 / len(methods)  # Width of each bar
    # group_width = bar_width * len(methods)  # Width of each group of bars

    # Set up positions for each bar within each dataset group
    group_positions = np.arange(len(datasets))

    # Create a function to generate method labels
    def get_method_label(method: str) -> str:
        if method.startswith("baseline_"):
            return f"Finetune: {method.split('_')[1]}"
        elif method.startswith("continuation_"):
            return f"Continue: {method.split('_')[1]}"
        elif method.startswith("probe_"):
            if "attention" in method:
                return "Attention Probe"
            elif "softmax" in method:
                return "Softmax Probe"
            else:
                return f"Probe: {method.split('_')[1]}"
        else:
            return method

    # Create handles for the legend
    legend_handles = []
    legend_labels = []

    for i, method in enumerate(methods):
        label = get_method_label(method)
        # Clean hatch pattern for this method
        hatch_pattern = hatch_dict.get(method, "")
        if hatch_pattern in ["o", "O"]:
            # Create a pattern with controlled density
            hatch_pattern = hatch_pattern * max(
                1, int(10 / hatchdensity_dict.get(method, 6))
            )

        # Create a patch for the legend
        import matplotlib.patches as mpatches

        patch = mpatches.Rectangle(
            (0, 0),
            1,
            1,
            fc=color_dict.get(method),
            hatch=hatch_pattern,
            alpha=0.8,
            edgecolor="black",
            linewidth=linewidth_dict.get(method, 0.5),
        )
        legend_handles.append(patch)
        legend_labels.append(label)

    # For each dataset, plot bars for each method
    for d, dataset in enumerate(datasets):
        # Filter data for this dataset
        dataset_data = plot_df[plot_df["dataset_name"] == dataset]

        # For each method, plot a bar
        for i, method in enumerate(methods):
            # Get data for this method
            method_data = dataset_data[dataset_data["method"] == method]

            if not method_data.empty:
                # Get metric value and error
                value = (
                    method_data[metric].values[0]
                    if not method_data[metric].isna().all()
                    else 0
                )
                error = (
                    method_data[f"{metric}_err"].values[0]
                    if not method_data[f"{metric}_err"].isna().all()
                    else None
                )

                # Clean hatch pattern for this method
                hatch_pattern = hatch_dict.get(method, "")
                if hatch_pattern in ["o", "O"]:
                    # Create a pattern with controlled density
                    hatch_pattern = hatch_pattern * max(
                        1, int(10 / hatchdensity_dict.get(method, 6))
                    )

                # Calculate position
                x_pos = group_positions[d] + (i - len(methods) / 2 + 0.5) * bar_width

                # Plot the bar
                ax.bar(
                    x_pos,
                    value,
                    width=bar_width,
                    color=color_dict.get(method),
                    hatch=hatch_pattern,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=linewidth_dict.get(method, 0.5),
                    yerr=error,
                    capsize=3,
                )

    # Set up x-axis
    ax.set_xticks(group_positions)
    ax.set_xticklabels(datasets, rotation=45, ha="right")

    # Add labels, title and legend
    ax.set_xlabel("Dataset")
    if metric == "auroc":
        ax.set_ylabel("AUROC")
    else:  # metric == "tpr_at_fpr"
        ax.set_ylabel("TPR at 1% FPR")

    # Legend is disabled by setting show_legend parameter to False by default

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits
    if metric == "auroc":
        ax.set_ylim(0.5, 1.0)
    else:  # metric == "tpr_at_fpr"
        ax.set_ylim(0.0, 1.0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.3)

    # Save the plot
    legend_suffix = "_nolegend"  # Always no legend
    output_filename = f"probes_vs_baseline_plot_{metric}{legend_suffix}.pdf"
    png_output_filename = f"probes_vs_baseline_plot_{metric}{legend_suffix}.png"
    plt.savefig(RESULTS_DIR / output_filename, bbox_inches="tight", dpi=600)
    plt.savefig(RESULTS_DIR / png_output_filename, bbox_inches="tight", dpi=600)
    plt.close()


def plot_combined_metrics(plot_df: pd.DataFrame) -> None:
    """
    Create a figure with two subplots showing mean AUROC and TPR@1%FPR across all datasets.
    One subplot for AUROC and another for TPR at 1% FPR.

    Args:
        plot_df: DataFrame containing the results to plot
    """
    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharey=False)

    # Define utility functions for model name and size extraction
    def extract_model_name(label: str) -> str:
        """Extract the base model name (e.g., llama, gemma) from the label."""
        import re

        # Look for common model names - adapt this regex as needed
        name_match = re.search(r"(?i)(llama|gemma|mistral|gpt|opt|phi)", label)
        if name_match:
            return name_match.group(1).lower()  # Convert to lowercase for consistency
        return ""  # Default if no model name found

    def extract_model_size(label: str) -> int:
        """Extract the model size (in billions) from the label."""
        import re

        size_match = re.search(r"(\d+)[bB]", label)
        if size_match:
            return int(size_match.group(1))
        return 0  # Default if no size found

    # Get all methods and categorize them
    all_methods = plot_df["method"].unique()

    # Separate methods by type
    probe_methods = sorted([m for m in all_methods if m.startswith("probe_")])
    finetuned_methods = sorted([m for m in all_methods if m.startswith("baseline_")])
    continuation_methods = sorted(
        [m for m in all_methods if m.startswith("continuation_")]
    )

    # Custom sort function to order models by name then size
    def sort_by_model_name_and_size(method_list: list[str]) -> list[str]:
        return sorted(
            method_list,
            key=lambda x: (
                extract_model_name(x.split("_")[1]),
                extract_model_size(x.split("_")[1]),
            ),
        )

    # Sort methods by model name and size
    probe_methods = sort_by_model_name_and_size(probe_methods)
    finetuned_methods = sort_by_model_name_and_size(finetuned_methods)
    continuation_methods = sort_by_model_name_and_size(continuation_methods)

    # Order methods with probes first, then finetuned, then continuations
    methods = probe_methods + finetuned_methods + continuation_methods

    # Define distinct colors for different method types
    probe_colors = {
        "probe_attention": "#ff7f0e",  # Orange for attention probe
        "probe_softmax": "#1f77b4",  # Blue for softmax probe
    }
    finetuned_colors = [
        "#32CD32",
        "#32CD32",
        "#006400",
        "#006400",
    ]
    continuation_colors = [
        "#00DEE1",
        "#00DEE1",
        "#00DEE1",
        (0.0, 0.5019607843137255, 0.5019607843137255),
        (0.0, 0.5019607843137255, 0.5019607843137255),
        (0.0, 0.5019607843137255, 0.5019607843137255),
    ]

    # Create a color dictionary
    color_dict = {}

    # Assign colors to probe methods based on whether they contain "attention" or "softmax"
    for method in probe_methods:
        if "attention" in method:
            color_dict[method] = probe_colors["probe_attention"]
        elif "softmax" in method:
            color_dict[method] = probe_colors["probe_softmax"]
        else:
            color_dict[method] = "#2ca02c"  # Default green for unrecognized probes

    # Assign colors to finetuned methods
    for i, method in enumerate(finetuned_methods):
        color_dict[method] = finetuned_colors[i % len(finetuned_colors)]

    # Assign colors to continuation methods
    for i, method in enumerate(continuation_methods):
        color_dict[method] = continuation_colors[i % len(continuation_colors)]

    # Define hatch patterns with variations for different methods and sizes
    hatch_patterns = {
        "small_1": "..",
        "small_3": "..",
        "medium_8": "o",
        "medium_12": "o",
        "large_27": "O",
        "large_70": "O",
    }

    # Store linewidth information for different model sizes
    linewidth_dict = {}
    # Store hatchdensity (spacing between elements) for different model sizes
    hatchdensity_dict = {}

    # Define hatch patterns based on model size
    hatch_dict = {}
    # No hatching for probe methods
    for method in probe_methods:
        hatch_dict[method] = ""
        linewidth_dict[method] = 0.5
        hatchdensity_dict[method] = 1

    # Assign hatches based on exact model size
    for method in finetuned_methods + continuation_methods:
        # Extract model name part after the prefix
        if method.startswith("baseline_"):
            model_name = method.split("_")[1]
        else:  # continuation
            model_name = method.split("_")[1]

        # Get model size
        size = extract_model_size(model_name)

        # Assign pattern based on exact size for specific thickness
        if size == 1:
            hatch_dict[method] = hatch_patterns["small_1"]
            # linewidth_dict[method] = 0.8
            # hatchdensity_dict[method] = 4  # More dense (many small dots)
        elif size <= 3:
            hatch_dict[method] = hatch_patterns["small_3"]
            # linewidth_dict[method] = 1.0
            hatchdensity_dict[method] = 6  # Dense small circles
        elif size <= 8:
            hatch_dict[method] = hatch_patterns["medium_8"]
            # linewidth_dict[method] = 1.2
            # hatchdensity_dict[method] = 8  # Medium spacing
        elif size <= 12 or "gemma-12b" in model_name.lower():
            hatch_dict[method] = hatch_patterns["medium_12"]
            # linewidth_dict[method] = 1.5
            # hatchdensity_dict[method] = 10  # Fewer medium circles
        elif size <= 27:
            hatch_dict[method] = hatch_patterns["large_27"]
            # linewidth_dict[method] = 1.8
            # hatchdensity_dict[method] = 12  # Few large circles
        else:  # > 27b (like 70b)
            hatch_dict[method] = hatch_patterns["large_70"]
            # linewidth_dict[method] = 2.0
            # hatchdensity_dict[method] = 15  # Very few very large circles

    # Calculate mean performance across all datasets for each method
    auroc_means = {}
    tpr_means = {}
    auroc_errors = {}
    tpr_errors = {}

    for method in methods:
        method_data = plot_df[plot_df["method"] == method]

        # Calculate means
        auroc_means[method] = method_data["auroc"].mean()
        tpr_means[method] = method_data["tpr_at_fpr"].mean()

        # Get the pre-calculated error values (95% confidence intervals)
        # These were calculated in create_plot_dataframe function
        auroc_errors[method] = (
            method_data["auroc_err"].mean()
            if not method_data["auroc_err"].isna().all()
            else None
        )
        tpr_errors[method] = (
            method_data["tpr_at_fpr_err"].mean()
            if not method_data["tpr_at_fpr_err"].isna().all()
            else None
        )

    # Set up the bar positions
    bar_width = 0.7
    total_methods = len(methods)
    x_positions = np.arange(total_methods)

    # Create a combined label for each method
    method_labels = []
    method_numbers = []
    for i, method in enumerate(methods):
        # Create numeric labels for x-axis
        method_numbers.append(str(i + 1))

        if method.startswith("baseline_"):
            method_label = f"Finetune: {method.split('_')[1]}"
        elif method.startswith("continuation_"):
            method_label = f"Continue: {method.split('_')[1]}"
        elif method.startswith("probe_"):
            if "attention" in method:
                method_label = "Attention Probe"
            elif "softmax" in method:
                method_label = "Softmax Probe"
            else:
                method_label = f"Probe: {method.split('_')[1]}"
        else:
            method_label = method
        method_labels.append(method_label)

    # AUROC bars on the first subplot
    for i, method in enumerate(methods):
        # Clean hatch pattern for this method
        hatch_pattern = hatch_dict.get(method, "")
        # if hatch_pattern in ["o", "O"]:
        #    # Create a pattern with controlled density
        #    hatch_pattern = hatch_pattern * max(
        #        1, int(10 / hatchdensity_dict.get(method, 6))
        #    )

        # Add error bars if available for this method
        yerr = auroc_errors.get(method, None)

        ax1.bar(
            x_positions[i],
            auroc_means[method],
            width=bar_width,
            color=color_dict.get(method),
            hatch=hatch_pattern,
            alpha=0.9,
            edgecolor="black",
            linewidth=linewidth_dict.get(method, 1),
            yerr=yerr,
            capsize=5,
        )

    # TPR@1%FPR bars on the second subplot
    for i, method in enumerate(methods):
        # Clean hatch pattern for this method
        hatch_pattern = hatch_dict.get(method, "")
        if hatch_pattern in ["o", "O"]:
            # Create a pattern with controlled density
            hatch_pattern = hatch_pattern * max(
                1, int(10 / hatchdensity_dict.get(method, 6))
            )

        # Add error bars if available for this method
        yerr = tpr_errors.get(method, None)

        ax2.bar(
            x_positions[i],
            tpr_means[method],
            width=bar_width,
            color=color_dict.get(method),
            hatch=hatch_pattern,
            alpha=0.9,
            edgecolor="black",
            linewidth=linewidth_dict.get(method, 1),
            yerr=yerr,
            capsize=5,
        )

    # Set up the axes
    # Remove the titles
    ax1.set_xlabel("Method", fontsize=12)
    ax2.set_xlabel("Method", fontsize=12)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax2.set_ylabel("TPR at 1% FPR", fontsize=12)

    # Set y-axis limits
    ax1.set_ylim(0.55, 1.0)
    ax2.set_ylim(0.0, 1.0)

    # Set tick positions and labels
    for ax in [ax1, ax2]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(method_numbers, rotation=0)
        ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.3)

    # Save the plot
    output_filename = "probes_vs_baseline_combined_metrics.pdf"
    png_output_filename = "probes_vs_baseline_combined_metrics.png"
    plt.savefig(RESULTS_DIR / output_filename, bbox_inches="tight", dpi=600)
    plt.savefig(RESULTS_DIR / png_output_filename, bbox_inches="tight", dpi=600)
    plt.close()


if __name__ == "__main__":
    probe_paths = [
        RESULTS_DIR / "probes/results_attention_test_1.jsonl",
        RESULTS_DIR / "probes/results_attention_test_2.jsonl",
        RESULTS_DIR / "probes/results_attention_test_3.jsonl",
        RESULTS_DIR / "probes/results_softmax_test_1.jsonl",
        RESULTS_DIR / "probes/results_softmax_test_2.jsonl",
        RESULTS_DIR / "probes/results_softmax_test_3.jsonl",
    ]

    finetune_paths = [
        RESULTS_DIR / "finetuned_baselines/finetuning_gemma_1b_test_optimized_0.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_gemma_1b_test_optimized_1.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_gemma_1b_test_optimized_2.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_gemma_12b_test.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_llama_1b_test_optimized_0.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_llama_1b_test_optimized_1.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_llama_1b_test_optimized_2.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_llama_8b_test_optimized_0.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_llama_8b_test_optimized_1.jsonl",
        RESULTS_DIR / "finetuned_baselines/finetuning_llama_8b_test_optimized_2.jsonl",
    ]

    continuation_paths = [
        RESULTS_DIR / "continuation_baselines/baseline_llama-1b_v2.jsonl",
        RESULTS_DIR / "continuation_baselines/baseline_gemma-1b_prompt_at_end.jsonl",
        RESULTS_DIR / "continuation_baselines/baseline_gemma-12b.jsonl",
        RESULTS_DIR / "continuation_baselines/baseline_gemma-27b.jsonl",
        RESULTS_DIR / "continuation_baselines/baseline_llama-70b.jsonl",
        RESULTS_DIR / "continuation_baselines/baseline_llama-8b_default.jsonl",
    ]

    df_combined = prepare_data(
        probe_paths=probe_paths,
        baseline_paths=finetune_paths,
        continuation_paths=continuation_paths,
    )

    df_plot = create_plot_dataframe(df_combined)

    # Plot with AUROC metric
    plot_results(df_plot, metric="auroc", show_legend=False)

    # Plot with TPR at 1% FPR metric
    plot_results(df_plot, metric="tpr_at_fpr", show_legend=False)

    # Plot the combined metrics chart
    plot_combined_metrics(df_plot)
