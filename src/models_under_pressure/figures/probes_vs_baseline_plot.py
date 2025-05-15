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
        tpr_at_fpr=("tpr_at_fpr", "mean"),
        tpr_at_fpr_std=("tpr_at_fpr", "std"),
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
    plot_df: pd.DataFrame, metric: str = "auroc", show_legend: bool = True
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

    # Custom sort function to swap "Aya Redteaming" and "MTS" and keep others in order
    def custom_dataset_sort(datasets_list):
        # Convert to list for manipulation
        datasets_list = list(datasets_list)

        # Find indices of Aya Redteaming and MTS
        aya_idx = next(
            (i for i, d in enumerate(datasets_list) if "Aya Redteaming" in d), -1
        )
        mts_idx = next((i for i, d in enumerate(datasets_list) if "MTS" in d), -1)

        # If both datasets are found, swap them
        if aya_idx != -1 and mts_idx != -1:
            datasets_list[aya_idx], datasets_list[mts_idx] = (
                datasets_list[mts_idx],
                datasets_list[aya_idx],
            )

        return datasets_list

    # Apply custom sort to datasets
    datasets = np.array(custom_dataset_sort(datasets))

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
        "limegreen",
        "limegreen",
        "green",
        "green",
    ]  # Shades of purple/pink for finetuned methods
    continuation_colors = [
        "darkturquoise",
        "darkturquoise",
        "teal",
        "teal",
    ]  # New colors for continuation methods

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
        "medium_12": "O",  # Larger circles for 12b
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
        elif size <= 12:
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

    # Calculate mean performance across all datasets for each method
    mean_performances = {}
    for method in methods:
        method_data = plot_df[plot_df["method"] == method]
        # Use appropriate metric for mean calculation
        if metric == "auroc":
            mean_performances[method] = method_data["auroc"].mean()
        else:  # metric == "tpr_at_fpr"
            mean_performances[method] = method_data["tpr_at_fpr"].mean()

    # Add "Mean" as the first position
    all_datasets = np.array(["Mean"] + list(datasets))

    # Set the x-tick positions using numeric values only
    positions = np.arange(len(all_datasets))
    ax.set_xticks(positions)

    # Don't set any text labels for the x-ticks
    ax.set_xticklabels([])

    # Optional: If you still want minimal numeric labels (just the numbers)
    for i, pos in enumerate(positions):
        ax.text(
            pos,
            -0.08,
            str(i),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
        )

    # Create method grouping information
    method_types = []
    for method in methods:
        if method.startswith("probe_"):
            method_types.append("probe")
        elif method.startswith("baseline_"):
            method_types.append("finetuned")
        elif method.startswith("continuation_"):
            method_types.append("continuation")
        else:
            method_types.append("other")

    # TODO: EDIT HERE
    # Calculate the total width needed for all bars including gaps
    gap_size = 0.03  # Smaller gap between method groups
    # total_bars = len(methods)

    # Get count of each method type
    probe_count = len(probe_methods)
    finetuned_count = len(finetuned_methods)
    continuation_count = len(continuation_methods)

    # TODO: EDIT HERE
    # Calculate bar width - use a fixed value that looks good 0.7
    bar_width = 0.95 / (
        probe_count + finetuned_count + continuation_count + 2
    )  # +2 for the gaps

    # Plot bars for each method
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df["method"] == method]

        # Track position relative to the group's start
        method_group_idx = 0

        # Calculate base position based on method type
        if method.startswith("probe_"):
            # Find position within probe group
            method_group_idx = probe_methods.index(method)
            group_start = 0
        elif method.startswith("baseline_"):
            # Find position within finetuned group
            method_group_idx = finetuned_methods.index(method)
            # Place finetuned after probes (original order)
            group_start = probe_count + gap_size / bar_width
        elif method.startswith("continuation_"):
            # Find position within continuation group
            method_group_idx = continuation_methods.index(method)
            # Place continuation after finetuned (original order)
            group_start = probe_count + finetuned_count + 2 * gap_size / bar_width

        # Calculate position with bar_width unit as the base
        relative_pos = group_start + method_group_idx

        # Position adjustment to center the bar groups
        # Calculate each position individually
        positions = []
        for j in range(len(all_datasets)):
            # Center everything around the x-tick position
            center = j
            # Total width of all bars and gaps
            total_width = (
                probe_count + finetuned_count + continuation_count
            ) * bar_width + 2 * gap_size
            # Start position (left edge of leftmost bar)
            start_pos = center - total_width / 2
            # Position of this specific bar
            pos = start_pos + relative_pos * bar_width
            positions.append(pos)

        # Create arrays aligned with all datasets (including Mean)
        values = np.zeros(len(all_datasets), dtype=float)
        values.fill(np.nan)
        errors = np.zeros(len(all_datasets), dtype=float)
        errors.fill(np.nan)

        # Set the mean value as the first position
        values[0] = mean_performances[method]

        # Fill in the values where we have data (starting from position 1)
        for idx, dataset in enumerate(datasets):
            dataset_data = method_data[method_data["dataset_name"] == dataset]
            if not dataset_data.empty:
                # Use the appropriate metric based on the flag
                if metric == "auroc":
                    values[idx + 1] = dataset_data["auroc"].iloc[0]
                    errors[idx + 1] = dataset_data["auroc_std"].iloc[0]
                else:  # metric == "tpr_at_fpr"
                    values[idx + 1] = dataset_data["tpr_at_fpr"].iloc[0]
                    errors[idx + 1] = dataset_data["tpr_at_fpr_std"].iloc[0]

        # Only use error bars where std is not NaN and not 0
        mask = (~np.isnan(errors)) & (errors > 0)
        yerr = np.zeros_like(errors)
        np.putmask(yerr, mask, errors)

        # Convert to list for matplotlib compatibility
        values_list = values.tolist()
        yerr_list = yerr.tolist()

        # Clean up labels for display
        if method.startswith("baseline_"):
            method_label = method[9:]  # Remove 'baseline_' prefix
            method_type = "finetuned"

            # Get hatch density (spacing) for this method
            hatch_density = hatchdensity_dict.get(method, 6)  # Default density

            # Create the combined hatching pattern with proper density
            hatch_pattern = hatch_dict.get(method, "")
            # For circular patterns, control the density based on model size
            if hatch_pattern in ["o", "O"]:
                # Create a pattern with controlled density
                hatch_pattern = hatch_pattern * max(1, int(10 / hatch_density))

        elif method.startswith("continuation_"):
            method_label = method[13:]  # Remove 'continuation_' prefix
            method_type = "continuation"

            # Get hatch density (spacing) for this method
            hatch_density = hatchdensity_dict.get(method, 6)  # Default density

            # Create the combined hatching pattern with proper density
            hatch_pattern = hatch_dict.get(method, "")
            # For circular patterns, control the density based on model size
            if hatch_pattern in ["o", "O"]:
                # Create a pattern with controlled density
                hatch_pattern = hatch_pattern * max(1, int(10 / hatch_density))

        elif method.startswith("probe_"):
            method_label = method[6:]  # Remove 'probe_' prefix
            method_type = "probe"

            # Get hatch density (spacing) for this method
            hatch_density = hatchdensity_dict.get(method, 6)  # Default density

            # Create the combined hatching pattern with proper density
            hatch_pattern = hatch_dict.get(method, "")
            # For circular patterns, control the density based on model size
            if hatch_pattern in ["o", "O"]:
                # Create a pattern with controlled density
                hatch_pattern = hatch_pattern * max(1, int(10 / hatch_density))
        else:
            method_label = method
            method_type = "other"

            # Get hatch density (spacing) for this method
            hatch_density = hatchdensity_dict.get(method, 6)  # Default density

            # Create the combined hatching pattern with proper density
            hatch_pattern = hatch_dict.get(method, "")
            # For circular patterns, control the density based on model size
            if hatch_pattern in ["o", "O"]:
                # Create a pattern with controlled density
                hatch_pattern = hatch_pattern * max(1, int(10 / hatch_density))

        # Use the method-specific line width when drawing the bar
        ax.bar(
            positions,
            values_list,
            width=bar_width,
            # Store method type in the label for later parsing
            label=f"{method_type}:{method_label}",
            color=color_dict.get(method),
            hatch=hatch_pattern,  # Use the size-based hatch pattern
            alpha=0.8,
            yerr=yerr_list if np.any(mask) else None,
            capsize=5,
            edgecolor="black",
            linewidth=linewidth_dict.get(method, 0.5),  # Use the size-based linewidth
        )

    # Add a dotted vertical line to separate Mean from individual datasets
    # Comment out or remove this line
    # ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.7, linewidth=1.5)

    # Add labels, title and legend
    ax.set_xlabel("Dataset")
    if metric == "auroc":
        ax.set_ylabel("AUROC")
    else:  # metric == "tpr_at_fpr"
        ax.set_ylabel("TPR at 1% FPR")

    # Force show_legend to always be False
    show_legend = False

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits
    if metric == "auroc":
        ax.set_ylim(0.55, 1.0)
    else:  # metric == "tpr_at_fpr"
        ax.set_ylim(0.0, 1.0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    legend_suffix = "" if show_legend else "_nolegend"
    output_filename = f"probes_vs_baseline_plot_{metric}{legend_suffix}.pdf"
    png_output_filename = f"probes_vs_baseline_plot_{metric}{legend_suffix}.png"
    plt.savefig(RESULTS_DIR / output_filename, bbox_inches="tight", dpi=300)
    plt.savefig(RESULTS_DIR / png_output_filename, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    probe_paths = [
        "/home/ucabwjn/models-under-pressure/data/results/evaluate_probes/results_attention_test_1.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/evaluate_probes/results_attention_test_2.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/evaluate_probes/results_attention_test_3.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/evaluate_probes/results_softmax_test_1.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/evaluate_probes/results_softmax_test_2.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/evaluate_probes/results_softmax_test_3.jsonl",
    ]

    finetune_paths = [
        "/home/ucabwjn/models-under-pressure/data/results/finetuned_baselines/finetuning_gemma_1b_test_1.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/finetuned_baselines/finetuning_gemma_1b_test_2.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/finetuned_baselines/finetuning_gemma_12b_test.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/finetuned_baselines/finetuning_llama_1b_test_1.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/finetuned_baselines/finetuning_llama_1b_test_2.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/finetuned_baselines/finetuning_llama_1b_test_3.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/finetuned_baselines/finetuning_llama-8b_test.jsonl",
    ]

    continuation_paths = [
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_gemma-12b_2.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_gemma-12b_3.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_gemma-27b_2.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_gemma-27b_3.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_llama-70b_2.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_llama-70b_3.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_llama-8b_v2.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_llama-8b_v3.jsonl",
        "/home/ucabwjn/models-under-pressure/data/results/continuation_baselines/baseline_llama-8b_v4.jsonl",
    ]

    df_combined = prepare_data(
        probe_paths=[Path(probe_path) for probe_path in probe_paths],
        baseline_paths=[Path(baseline_path) for baseline_path in finetune_paths],
        continuation_paths=[Path(contin_path) for contin_path in continuation_paths],
    )

    df_plot = create_plot_dataframe(df_combined)

    # Plot with AUROC metric
    plot_results(df_plot, metric="auroc", show_legend=True)

    # Plot with TPR at 1% FPR metric
    plot_results(df_plot, metric="tpr_at_fpr", show_legend=False)
