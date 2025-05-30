from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import RESULTS_DIR
from models_under_pressure.experiments.evaluate_probes import calculate_metrics
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


def calculate_metric(
    group: pd.DataFrame, metric: str = "auroc", fpr: float = 0.01
) -> float:
    """Calculate the specified metric for a group of data.

    Args:
        group: DataFrame containing the data
        metric: Metric to calculate, either "auroc" or "tpr_at_fpr"
        fpr: False positive rate threshold for TPR calculation (default: 0.01)

    Returns:
        The calculated metric value
    """
    y_true = group["ground_truth_labels"].to_numpy()
    y_pred = group["scores"].to_numpy()

    metrics = calculate_metrics(np.array(y_true), np.array(y_pred), fpr=fpr)

    if metric == "auroc":
        return metrics["auroc"]
    elif metric == "tpr_at_fpr":
        return metrics["tpr_at_fpr"]
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def create_plot_dataframe(
    df: pd.DataFrame, metric: str = "auroc", fpr: float = 0.01
) -> pd.DataFrame:
    """
    Create the dataframe that will eventually be used to create the plot.

    Args:
        df: Input DataFrame containing the data
        metric: Metric to calculate, either "auroc" or "tpr_at_fpr"
        fpr: False positive rate threshold for TPR calculation (default: 0.01)
    """
    # Filter out MTS dataset for TPR at FPR metric
    # if metric == "tpr_at_fpr":
    #    df = df[df["dataset_name"] != "MTS"]

    # First calculate the metric for each dataset, method, and load_id
    grp = (
        df.groupby(["dataset_name", "method", "load_id"])
        .apply(lambda x: calculate_metric(x, metric, fpr))
        .reset_index()
        .rename(columns={0: metric})
    )

    # Calculate mean and std across load_ids for each dataset and method
    plot_df = (
        grp.groupby(["dataset_name", "method"])
        .agg(
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
        )
        .reset_index()
    )

    # Calculate mean across datasets for each method
    method_means = plot_df.groupby("method")["metric_mean"].mean().reset_index()
    method_means["dataset_name"] = "Mean"
    method_means["metric_std"] = 0  # No std for mean across datasets

    # For TPR at FPR, filter out MTS from the plot but keep it in the mean
    if metric == "tpr_at_fpr":
        plot_df = plot_df[plot_df["dataset_name"] != "MTS"]

    # Combine the dataset-specific results with the method means
    plot_df = pd.concat([method_means, plot_df], ignore_index=True)

    return plot_df


# Global variable for method name mapping
METHOD_NAME_MAPPING = {
    "attention": "Attention",
    "linear_then_softmax": "Softmax",
    "gemma-3-1b-it": "Gemma-3-1B",
    "gemma-3-12b-it": "Gemma-3-12B",
    "gemma-3-27b-it": "Gemma-3-27B",
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
}


def plot_results(
    plot_df: pd.DataFrame,
    metric: str = "auroc",
    fpr: float = 0.01,
    fontsize: int = 13,
    show_legend: bool = True,
) -> None:
    """
    Plot the results as a grouped bar chart with error bars where available.
    Probe methods are plotted first with distinct colors.
    Finetuned and continuation methods share colors but have different patterns.

    Args:
        plot_df: DataFrame containing the plot data
        metric: Metric to plot, either "auroc" or "tpr_at_fpr"
        fpr: False positive rate threshold for TPR calculation (default: 0.01)
        fontsize: Font size for the plot elements (default: 13)
        show_legend: Whether to display the legend (default: True)
    """
    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context(
        "paper", font_scale=fontsize / 10
    )  # Adjust font_scale based on fontsize

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
    all_methods = plot_df["method"].unique()

    # Separate methods by type
    probe_methods = sorted([m for m in all_methods if m.startswith("probe_")])
    finetuned_methods = sorted([m for m in all_methods if m.startswith("baseline_")])
    continuation_methods = sorted(
        [m for m in all_methods if m.startswith("continuation_")]
    )

    # Sort finetuned and continuation methods by model name and size
    finetuned_methods = sorted(
        finetuned_methods, key=lambda x: (extract_model_name(x), extract_model_size(x))
    )
    continuation_methods = sorted(
        continuation_methods,
        key=lambda x: (extract_model_name(x), extract_model_size(x)),
    )

    # Order methods with probes first, then finetuned, then continuations
    methods = probe_methods + finetuned_methods + continuation_methods

    # Define distinct colors for different method types
    # Use predefined colors for simplicity
    probe_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Define colors for model families and types
    finetuned_llama_color = "#008000"
    finetuned_gemma_color = "#32CD32"
    continuation_llama_color = "#008080"
    continuation_gemma_color = "#00CED1"

    # Create a color dictionary
    color_dict = {}
    for i, method in enumerate(probe_methods):
        color_dict[method] = probe_colors[i % len(probe_colors)]

    # Assign colors based on model family and type
    for method in finetuned_methods:
        if "llama" in method.lower():
            color_dict[method] = finetuned_llama_color
        elif "gemma" in method.lower():
            color_dict[method] = finetuned_gemma_color

    for method in continuation_methods:
        if "llama" in method.lower():
            color_dict[method] = continuation_llama_color
        elif "gemma" in method.lower():
            color_dict[method] = continuation_gemma_color

    # Define hatch patterns - using the same patterns for finetuned and continuation methods with same index
    hatch_dict = {}
    for method in probe_methods:
        hatch_dict[method] = ""  # No hatch for probes

    # Assign hatches based on model size
    for method in finetuned_methods + continuation_methods:
        model_size = extract_model_size(method)
        if model_size <= 1:
            hatch_dict[method] = "/"
        elif 1 < model_size <= 15:
            hatch_dict[method] = "x"
        else:
            hatch_dict[method] = "-"

    # Calculate mean performance across all datasets for each method
    mean_performances = {}
    for method in methods:
        method_data = plot_df[plot_df["method"] == method]
        mean_performances[method] = method_data[method_data["dataset_name"] == "Mean"][
            "metric_mean"
        ].iloc[0]

    # Add "Mean" as the first position
    all_datasets = np.array(["Mean"] + list(datasets[datasets != "Mean"]))

    # Set up bar positions
    x = np.arange(len(all_datasets))

    # Calculate the number of methods in each category
    n_probe_methods = len(probe_methods)
    n_finetuned_methods = len(finetuned_methods)

    # Calculate total width needed for each category including gaps
    total_width = 0.8  # Total width available for all bars
    gap_size = 0.03  # Reduced gap size between categories in bar width units
    bar_width = (total_width - 2 * gap_size) / len(
        methods
    )  # Adjust bar width based on number of methods

    # Calculate widths for each category
    probe_width = (total_width - 2 * gap_size) * (n_probe_methods / len(methods))
    finetuned_width = (total_width - 2 * gap_size) * (
        n_finetuned_methods / len(methods)
    )

    # Plot bars for each method
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df["method"] == method]

        # Calculate position based on method type
        if method.startswith("probe_"):
            # Position within probe section
            idx = probe_methods.index(method)
            position = x - total_width / 2 + idx * bar_width
        elif method.startswith("baseline_"):
            # Position within finetuned section
            idx = finetuned_methods.index(method)
            position = x - total_width / 2 + probe_width + gap_size + idx * bar_width
        else:  # continuation methods
            # Position within continuation section
            idx = continuation_methods.index(method)
            position = (
                x
                - total_width / 2
                + probe_width
                + finetuned_width
                + 2 * gap_size
                + idx * bar_width
            )

        # Create arrays aligned with all datasets (including Mean)
        values = np.zeros(len(all_datasets), dtype=float)
        values.fill(np.nan)
        errors = np.zeros(len(all_datasets), dtype=float)
        errors.fill(np.nan)

        # Set the mean value as the first position
        values[0] = mean_performances[method]

        # Fill in the values where we have data (starting from position 1)
        for idx, dataset in enumerate(datasets[datasets != "Mean"]):
            dataset_data = method_data[method_data["dataset_name"] == dataset]
            if not dataset_data.empty:
                values[idx + 1] = dataset_data["metric_mean"].iloc[0]
                errors[idx + 1] = dataset_data["metric_std"].iloc[0]

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
        elif method.startswith("continuation_"):
            method_label = method[13:]  # Remove 'continuation_' prefix
            method_type = "continuation"
        elif method.startswith("probe_"):
            method_label = method[6:]  # Remove 'probe_' prefix
            method_type = "probe"
        else:
            method_label = method
            method_type = "other"

        # Use the mapping for the label if available, otherwise use the original label
        display_label = METHOD_NAME_MAPPING.get(method_label, method_label)

        ax.bar(
            position,
            values_list,
            width=bar_width,
            # Store method type in the label for later parsing
            label=f"{method_type}:{display_label}",
            color=color_dict.get(method),
            hatch=hatch_dict.get(method),
            alpha=0.8,
            yerr=yerr_list if np.any(mask) else None,
            capsize=5,
            edgecolor="black",
            linewidth=0.5,
        )

    # Set the x-tick positions
    ax.set_xticks(x)

    # Create bold "Mean" and regular dataset labels
    ticklabels = [r"$\mathbf{Mean}$"] + list(datasets[datasets != "Mean"])
    ax.set_xticklabels(
        ticklabels, ha="center", fontsize=fontsize
    )  # Added fontsize for x-tick labels

    # Add a dotted vertical line to separate Mean from individual datasets
    ax.axvline(x=0.45, color="black", linestyle="--", alpha=0.6, linewidth=2.0)

    # Add y-tick labels with fontsize
    ax.tick_params(axis="y", labelsize=fontsize)

    # Add labels, title and legend
    ax.set_xlabel("Dataset", fontsize=fontsize)
    if metric == "auroc":
        ax.set_ylabel("AUROC", fontsize=fontsize)
    else:
        ax.set_ylabel(f"TPR at {int(fpr * 100)}% FPR", fontsize=fontsize)

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

    if show_legend:
        # Create the legend with the sequential structure
        legend = ax.legend(
            legend_handles,
            legend_labels,
            ncol=3,  # Three columns
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            fontsize=fontsize,
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
                legend_texts[i].set_fontsize(fontsize)

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits based on metric
    if metric == "auroc":
        ax.set_ylim(0.55, 1.0)
    else:  # tpr_at_fpr
        ax.set_ylim(0.0, 1.0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        # RESULTS_DIR / "probes_vs_baseline_plot.png", bbox_inches="tight", dpi=300
        RESULTS_DIR / f"probes_vs_baseline_plot_{metric}.pdf",
        bbox_inches="tight",
    )
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
        RESULTS_DIR / "continuation_baselines/baseline_llama-8b_default.jsonl",
        RESULTS_DIR / "continuation_baselines/baseline_gemma-12b.jsonl",
        RESULTS_DIR / "continuation_baselines/baseline_gemma-27b.jsonl",
        RESULTS_DIR / "continuation_baselines/baseline_llama-70b.jsonl",
    ]

    df_combined = prepare_data(
        probe_paths=probe_paths,
        baseline_paths=finetune_paths,
        continuation_paths=continuation_paths,
    )

    # Calculate metrics for each dataset
    # metric = "tpr_at_fpr"  # or "auroc"
    metric = "auroc"
    fpr = 0.01
    df_plot = create_plot_dataframe(df_combined, metric=metric, fpr=fpr)
    plot_results(df_plot, metric=metric, fpr=fpr, fontsize=18, show_legend=True)
