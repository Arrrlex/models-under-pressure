from pathlib import Path

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

    grp = (
        df.groupby(["dataset_name", "method", "load_id"])
        .apply(lambda x: float(roc_auc_score(x["ground_truth_labels"], x["scores"])))
        .reset_index()
        .rename(columns={0: "auroc"})
    )

    # Now group by the dataset_name and method and calculate the mean auroc and std of the auroc column:
    plot_df = grp.groupby(["dataset_name", "method"]).agg(
        auroc=("auroc", "mean"),
        auroc_std=("auroc", "std"),
    )

    return plot_df.reset_index()


def plot_results(plot_df: pd.DataFrame) -> None:
    """
    Plot the results as a grouped bar chart with error bars where available.
    Probe methods are plotted first with distinct colors.
    Finetuned and continuation methods share colors but have different patterns.
    """

    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

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
    all_methods = plot_df["method"].unique()

    # Separate methods by type
    probe_methods = sorted([m for m in all_methods if m.startswith("probe_")])
    finetuned_methods = sorted([m for m in all_methods if m.startswith("baseline_")])
    continuation_methods = sorted(
        [m for m in all_methods if m.startswith("continuation_")]
    )

    # Order methods with probes first, then finetuned, then continuations
    methods = probe_methods + finetuned_methods + continuation_methods

    # Define distinct colors for different method types
    # Use predefined colors for simplicity
    probe_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    finetuned_color = "brown"  # Brown for all finetuned methods
    continuation_color = "green"  # Green for all continuation methods

    # Create a color dictionary
    color_dict = {}
    for i, method in enumerate(probe_methods):
        color_dict[method] = probe_colors[i % len(probe_colors)]
    for method in finetuned_methods:
        color_dict[method] = finetuned_color
    for method in continuation_methods:
        color_dict[method] = continuation_color

    # Define hatch patterns with variations for different methods
    hatch_patterns = ["/", "\\", "x", "+", "*", "o", "O", ".", "|", "-"]

    # Define hatch patterns - using the same patterns for finetuned and continuation methods with same index
    hatch_dict = {}
    for method in probe_methods:
        hatch_dict[method] = ""  # No hatch for probes

    # Create mappings to match finetuned and continuation methods by model name
    finetuned_name_to_index = {
        method.split("_")[1]: i for i, method in enumerate(finetuned_methods)
    }
    continuation_name_to_index = {
        method.split("_")[1]: i for i, method in enumerate(continuation_methods)
    }

    # Assign hatches to finetuned methods
    for i, method in enumerate(finetuned_methods):
        hatch_dict[method] = hatch_patterns[i % len(hatch_patterns)]

    # Assign the same hatch to continuation methods with matching model names when possible
    for method in continuation_methods:
        model_name = method.split("_")[1]
        # Try to find matching finetuned to use same pattern, or fall back to index-based assignment
        if model_name in finetuned_name_to_index:
            finetuned_index = finetuned_name_to_index[model_name]
            hatch_dict[method] = hatch_patterns[finetuned_index % len(hatch_patterns)]
        else:
            # Fallback if no matching finetuned exists
            i = continuation_name_to_index[model_name]
            hatch_dict[method] = hatch_patterns[i % len(hatch_patterns)]

    # Calculate mean performance across all datasets for each method
    mean_performances = {}
    for method in methods:
        method_data = plot_df[plot_df["method"] == method]
        mean_performances[method] = method_data["auroc"].mean()

    # Add "Mean" as the first position
    all_datasets = np.array(["Mean"] + list(datasets))

    # Set up bar positions
    bar_width = 0.8 / len(methods)  # Adjust bar width based on number of methods
    x = np.arange(len(all_datasets))

    # Plot bars for each method
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df["method"] == method]

        # Position adjustment to center the bar groups
        position = x + bar_width * (i - len(methods) / 2 + 0.5)

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
                values[idx + 1] = dataset_data["auroc"].iloc[0]
                errors[idx + 1] = dataset_data["auroc_std"].iloc[0]

        # Only use error bars where std is not NaN and not 0
        mask = (~np.isnan(errors)) & (errors > 0)
        yerr = np.zeros_like(errors)
        np.putmask(yerr, mask, errors)

        # Convert to list for matplotlib compatibility
        values_list = values.tolist()
        yerr_list = yerr.tolist()

        # Plot the bars with error bars
        # Clean up labels for display
        if method.startswith("baseline_"):
            method_label = method[9:]  # Remove 'baseline_' prefix
            method_type = "finetuned"

            # Use model size to determine hatch density for finetuned models
            model_size = extract_model_size(method_label)

            # Assign hatch patterns based on model size
            if model_size <= 1:
                hatch_pattern = "/"  # Forward slash for small models
            elif model_size <= 10:
                hatch_pattern = "\\"  # Backslash for medium models
            else:
                hatch_pattern = "x"  # Combined slashes for large models

        elif method.startswith("continuation_"):
            method_label = method[13:]  # Remove 'continuation_' prefix
            method_type = "continuation"

            # Use model size to determine hatch density for continuation models
            model_size = extract_model_size(method_label)

            # Assign hatch patterns based on model size
            if model_size <= 10:
                hatch_pattern = "."  # Sparse dots for small models
            elif model_size <= 30:
                hatch_pattern = ".."  # Medium density dots for medium models
            else:
                hatch_pattern = "..."  # Dense dots for large models

        elif method.startswith("probe_"):
            method_label = method[6:]  # Remove 'probe_' prefix
            method_type = "probe"
            hatch_pattern = ""  # No hatch for probes
        else:
            method_label = method
            method_type = "other"
            hatch_pattern = ""

        ax.bar(
            position,
            values_list,
            width=bar_width,
            # Store method type in the label for later parsing
            label=f"{method_type}:{method_label}",
            color=color_dict.get(method),
            hatch=hatch_pattern,  # Use the density-based hatch pattern
            alpha=0.8,
            yerr=yerr_list if np.any(mask) else None,
            capsize=5,
            edgecolor="black",
            linewidth=0.5,
        )

    # Set the x-tick positions
    ax.set_xticks(x)

    # Create bold "Mean" and regular dataset labels
    ticklabels = [r"$\mathbf{Mean}$"] + list(datasets)
    ax.set_xticklabels(ticklabels, rotation=45, ha="right")

    # Add a dotted vertical line to separate Mean from individual datasets
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.7, linewidth=1.5)

    # Add labels, title and legend
    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUROC")

    # Add a legend with appropriate grouping by method type
    handles, labels = ax.get_legend_handles_labels()

    # Separate handles and labels by method type
    probe_handles = []
    probe_labels = []
    finetuned_handles = []
    finetuned_labels = []
    continuation_handles = []
    continuation_labels = []

    # Group handles and labels by method type
    for handle, label in zip(handles, labels):
        method_type, actual_label = label.split(":", 1)
        if method_type == "probe":
            probe_handles.append(handle)
            probe_labels.append(actual_label)
        elif method_type == "finetuned":
            finetuned_handles.append(handle)
            finetuned_labels.append(actual_label)
        elif method_type == "continuation":
            continuation_handles.append(handle)
            continuation_labels.append(actual_label)

    # Sort finetuned and continuation results by model name and then size
    finetuned_pairs = list(zip(finetuned_handles, finetuned_labels))
    finetuned_pairs.sort(
        key=lambda pair: (extract_model_name(pair[1]), extract_model_size(pair[1]))
    )
    finetuned_handles, finetuned_labels = (
        zip(*finetuned_pairs) if finetuned_pairs else ([], [])
    )

    continuation_pairs = list(zip(continuation_handles, continuation_labels))
    continuation_pairs.sort(
        key=lambda pair: (extract_model_name(pair[1]), extract_model_size(pair[1]))
    )
    continuation_handles, continuation_labels = (
        zip(*continuation_pairs) if continuation_pairs else ([], [])
    )

    # Convert back to lists if they became tuples from zip(*...)
    finetuned_handles = list(finetuned_handles)
    finetuned_labels = list(finetuned_labels)
    continuation_handles = list(continuation_handles)
    continuation_labels = list(continuation_labels)

    # Clear existing legends if any
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # Make space at the bottom for the legend
    plt.subplots_adjust(bottom=0.25)

    # Find max number of items in each category
    max_probe_items = len(probe_handles)
    max_finetuned_items = len(finetuned_handles)
    max_continuation_items = len(continuation_handles)

    # Create invisible handles for the column headers
    from matplotlib.patches import Patch

    probe_title = Patch(color="white", alpha=0)
    finetuned_title = Patch(color="white", alpha=0)
    continuation_title = Patch(color="white", alpha=0)

    # Add empty patches as spacers (completely transparent)
    empty = Patch(color="white", alpha=0)

    # Create legend handles and labels lists as requested
    # Format: [TITLE1, result1_1, result1_2, ..., filler, TITLE2, result2_1, ...]
    legend_handles = []
    legend_labels = []

    # Track title indices for later styling
    title_indices = []

    # Add PROBES title and results
    title_indices.append(len(legend_handles))  # Track this title's index
    legend_handles.append(probe_title)
    legend_labels.append("PROBES")

    # Add probe results
    legend_handles.extend(probe_handles)
    legend_labels.extend(probe_labels)

    # Add fillers to reach maximum length if needed
    for i in range(
        max_probe_items,
        max(max_probe_items, max_finetuned_items, max_continuation_items),
    ):
        legend_handles.append(empty)
        legend_labels.append("")

    # Add FINETUNED title and results
    title_indices.append(len(legend_handles))  # Track this title's index
    legend_handles.append(finetuned_title)
    legend_labels.append("FINETUNED")

    # Add finetuned results
    legend_handles.extend(finetuned_handles)
    legend_labels.extend(finetuned_labels)

    # Add fillers to reach maximum length if needed
    for i in range(
        max_finetuned_items,
        max(max_probe_items, max_finetuned_items, max_continuation_items),
    ):
        legend_handles.append(empty)
        legend_labels.append("")

    # Add CONTINUATION title and results
    title_indices.append(len(legend_handles))  # Track this title's index
    legend_handles.append(continuation_title)
    legend_labels.append("CONTINUATION")

    # Add continuation results
    legend_handles.extend(continuation_handles)
    legend_labels.extend(continuation_labels)

    # No need to add fillers for the last category

    # Create the legend with the sequential structure
    legend = ax.legend(
        legend_handles,
        legend_labels,
        ncol=3,  # Three columns
        loc="lower left",
        # bbox_to_anchor=(0.5, -0.15),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=13,  # Increased from 9 to 11
        columnspacing=1.0,
        handletextpad=0.5,
        handlelength=1.5,
        borderpad=0.7,
        shadow=True,
    )

    # Apply bold formatting only to titles using the tracked indices
    legend_texts = legend.get_texts()
    for i in title_indices:
        if i < len(legend_texts):
            legend_texts[i].set_fontweight("bold")
            legend_texts[i].set_fontsize(13)  # Increased from 10 to 13 for titles

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits
    ax.set_ylim(0.55, 1.0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        RESULTS_DIR / "probes_vs_baseline_plot.png", bbox_inches="tight", dpi=300
    )
    plt.close()


if __name__ == "__main__":
    probe_paths = [
        "/home/xxx/models-under-pressure/data/results/evaluate_probes/results_attention_test_1.jsonl",
        "/home/xxx/models-under-pressure/data/results/evaluate_probes/results_attention_test_2.jsonl",
        "/home/xxx/models-under-pressure/data/results/evaluate_probes/results_attention_test_3.jsonl",
        "/home/xxx/models-under-pressure/data/results/evaluate_probes/results_softmax_test_1.jsonl",
        "/home/xxx/models-under-pressure/data/results/evaluate_probes/results_softmax_test_2.jsonl",
        "/home/xxx/models-under-pressure/data/results/evaluate_probes/results_softmax_test_3.jsonl",
    ]

    finetune_paths = [
        "/home/xxx/models-under-pressure/data/results/finetuned_baselines/finetuning_gemma_1b_test_1.jsonl",
        "/home/xxx/models-under-pressure/data/results/finetuned_baselines/finetuning_gemma_1b_test_2.jsonl",
        "/home/xxx/models-under-pressure/data/results/finetuned_baselines/finetuning_gemma_12b_test.jsonl",
        "/home/xxx/models-under-pressure/data/results/finetuned_baselines/finetuning_llama_1b_test_1.jsonl",
        "/home/xxx/models-under-pressure/data/results/finetuned_baselines/finetuning_llama_1b_test_2.jsonl",
        "/home/xxx/models-under-pressure/data/results/finetuned_baselines/finetuning_llama_1b_test_3.jsonl",
        "/home/xxx/models-under-pressure/data/results/finetuned_baselines/finetuning_llama-8b_test.jsonl",
    ]

    continuation_paths = [
        "/home/xxx/models-under-pressure/data/results/continuation_baselines/baseline_gemma-12b_2.jsonl",
        "/home/xxx/models-under-pressure/data/results/continuation_baselines/baseline_gemma-12b_3.jsonl",
        "/home/xxx/models-under-pressure/data/results/continuation_baselines/baseline_gemma-27b_2.jsonl",
        "/home/xxx/models-under-pressure/data/results/continuation_baselines/baseline_gemma-27b_3.jsonl",
        "/home/xxx/models-under-pressure/data/results/continuation_baselines/baseline_llama-70b_2.jsonl",
        "/home/xxx/models-under-pressure/data/results/continuation_baselines/baseline_llama-70b_3.jsonl",
    ]

    df_combined = prepare_data(
        probe_paths=[Path(probe_path) for probe_path in probe_paths],
        baseline_paths=[Path(baseline_path) for baseline_path in finetune_paths],
        continuation_paths=[Path(contin_path) for contin_path in continuation_paths],
    )

    df_plot = create_plot_dataframe(df_combined)

    # Calculate AUROC for each dataset
    plot_results(df_plot)
