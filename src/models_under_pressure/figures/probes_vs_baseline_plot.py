from pathlib import Path

import pandas as pd

from models_under_pressure.config import RESULTS_DIR
from models_under_pressure.figures.utils import (
    get_baseline_results,
    get_continuation_results,
    get_probe_results,
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
        columns={"output_scores": "probe_scores", "output_labels": "probe_labels"},
        inplace=True,
    )
    baseline_results.rename(
        columns={
            "scores": "baseline_scores",
        },
        inplace=True,
    )
    continuation_results.rename(
        columns={
            "high_stakes_scores": "continuation_scores",
        },
        inplace=True,
    )

    # Add prefix to probe columns
    probe_cols = [
        col
        for col in probe_results.columns
        if col
        not in [
            "probe_scores",
            "probe_labels",
            "probe_name",
            "probe_spec",
            "dataset_name",
            "ids",
        ]
    ]
    probe_results.rename(
        columns={col: f"probe_{col}" for col in probe_cols}, inplace=True
    )

    # Add prefix to baseline columns
    baseline_cols = [
        col
        for col in baseline_results.columns
        if col not in ["baseline_scores", "dataset_name", "ids"]
    ]
    baseline_results.rename(
        columns={col: f"baseline_{col}" for col in baseline_cols}, inplace=True
    )

    # Add prefix to continuation columns
    continuation_cols = [
        col
        for col in continuation_results.columns
        if col not in ["continuation_scores", "dataset_name", "ids"]
    ]
    continuation_results.rename(
        columns={col: f"continuation_{col}" for col in continuation_cols}, inplace=True
    )

    # Combine the results on the ids and dataset_name columns
    df_combined_results = pd.merge(
        probe_results, baseline_results, on=["ids", "dataset_name"], how="inner"
    )

    df_combined_results = pd.merge(
        df_combined_results,
        continuation_results,
        on=["ids", "dataset_name"],
        how="inner",
    )

    return df_combined_results


def create_plot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the dataframe that will eventually be used to create the plot.

    """

    # Create the individual probe, continuation and baseline results:

    # Aggregate by dataset_name and calculate the AUROC score for the probe_scores, baseline_scores and continuation_scores vs ground truth labels
    from sklearn.metrics import roc_auc_score

    # Define functions to calculate AUROC for each method
    def calc_probe_auroc(group):
        return roc_auc_score(group["probe_ground_truth_labels"], group["probe_scores"])

    def calc_baseline_auroc(group):
        return roc_auc_score(group["baseline_ground_truth"], group["baseline_scores"])

    def calc_continuation_auroc(group):
        return roc_auc_score(
            group["continuation_ground_truth"], group["continuation_scores"]
        )

    # Calculate AUROC scores using groupby and agg
    plot_df = (
        df.groupby("dataset_name")
        .agg(
            probe_auroc=("dataset_name", lambda x: calc_probe_auroc(df.loc[x.index])),
            baseline_auroc=(
                "dataset_name",
                lambda x: calc_baseline_auroc(df.loc[x.index]),
            ),
            continuation_auroc=(
                "dataset_name",
                lambda x: calc_continuation_auroc(df.loc[x.index]),
            ),
        )
        .reset_index()
    )

    # Melt the dataframe to get it into the right format for plotting
    plot_df = pd.melt(
        plot_df,
        id_vars=["dataset_name"],
        value_vars=["probe_auroc", "baseline_auroc", "continuation_auroc"],
        var_name="method",
        value_name="auroc",
    )

    # Clean up method names by removing '_auroc' suffix
    plot_df["method"] = plot_df["method"].str.replace("_auroc", "")

    return plot_df


def plot_results(plot_df: pd.DataFrame) -> None:
    """
    Plot the results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for each method
    colors = {
        "probe": "#1f77b4",  # blue
        "baseline": "#ff7f0e",  # orange
        "continuation": "#2ca02c",  # green
    }

    # Create the bar plot
    bar_width = 0.25
    datasets = plot_df["dataset_name"].unique()
    x = np.arange(len(datasets))

    # Plot bars for each method
    for i, method in enumerate(["probe", "baseline", "continuation"]):
        method_data = plot_df[plot_df["method"] == method]
        ax.bar(
            x + (i - 1) * bar_width,
            method_data["auroc"],
            width=bar_width,
            label=method.capitalize(),
            color=colors[method],
            alpha=0.8,
        )

    # Add labels, title and legend
    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUROC")
    ax.set_title("Performance Comparison: Probe vs Baseline vs Continuation")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for i, method in enumerate(["probe", "baseline", "continuation"]):
        method_data = plot_df[plot_df["method"] == method]
        for j, value in enumerate(method_data["auroc"]):
            ax.text(
                x[j] + (i - 1) * bar_width,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )

    # Set y-axis limits
    ax.set_ylim(0, 1.1)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Save the plot
    plt.savefig(RESULTS_DIR / "probes_vs_baseline_plot.png")


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
    plot_results(df_plot)
