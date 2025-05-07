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

    continuation_results["method"] = (
        "continuation_" + continuation_results["model_name"]
    )
    baseline_results["method"] = "baseline_" + baseline_results["model_name"]
    probe_results["method"] = "probe_" + probe_results["probe_name"]

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
    """

    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Define colors for each method
    colors = {
        "probe_attention": "#1f77b4",  # blue
        "baseline_meta-llama/Llama-3.1-8B-Instruct": "#ff7f0e",  # orange
        "continuation_meta-llama/Llama-3.3-70B-Instruct": "#2ca02c",  # green
    }

    # Get unique datasets and methods
    datasets = plot_df["dataset_name"].unique()
    methods = plot_df["method"].unique()

    # Calculate mean performance across all datasets for each method
    mean_performances = {}
    for method in methods:
        method_data = plot_df[plot_df["method"] == method]
        mean_performances[method] = method_data["auroc"].mean()

    # Add "Mean" as the first position
    all_datasets = np.array(["Mean"] + list(datasets))

    # Set up bar positions
    bar_width = 0.25
    x = np.arange(len(all_datasets))

    # Plot bars for each method
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df["method"] == method]

        # Create arrays aligned with all datasets (including Mean)
        values = np.full(len(all_datasets), np.nan)
        errors = np.full(len(all_datasets), np.nan)

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
        yerr = np.where(mask, errors, 0)

        # Plot the bars with error bars
        ax.bar(
            x + (i - 1) * bar_width,
            values,
            width=bar_width,
            label=method,
            color=colors.get(
                method, f"C{i}"
            ),  # Use predefined color or fallback to default
            alpha=0.8,
            yerr=yerr
            if np.any(mask)
            else None,  # Only show error bars if we have any valid errors
            capsize=5,
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
    ax.set_title("Performance Comparison Across Datasets")

    ax.legend(ncols=3, loc="upper left")

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits
    ax.set_ylim(0, 1.1)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        RESULTS_DIR / "probes_vs_baseline_plot.png", bbox_inches="tight", dpi=300
    )
    plt.close()


if __name__ == "__main__":
    baseline_path = "/home/ubuntu/models-under-pressure/data/results/finetune_baselines_Llama-3.1-8B-Instruct__0_2025-05-06.jsonl"
    probe_path = "/home/ubuntu/models-under-pressure/data/results/evaluate_probes/results_wOgjTcLk_20250505_094202.jsonl"
    contin_path = (
        "/home/ubuntu/models-under-pressure/data/results/baseline_llama-70b.jsonl"
    )

    df_combined = prepare_data(
        probe_paths=[Path(probe_path)],
        baseline_paths=[Path(baseline_path)],
        continuation_paths=[Path(contin_path)],
    )

    df_plot = create_plot_dataframe(df_combined)

    # Calculate AUROC for each dataset
    plot_results(df_plot)
