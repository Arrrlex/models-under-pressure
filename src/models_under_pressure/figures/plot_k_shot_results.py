from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models_under_pressure.config import EVALUATE_PROBES_DIR, PLOTS_DIR
from models_under_pressure.interfaces.results import KShotResult

# Set style parameters
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14,
    }
)

sns.set_style("whitegrid")


def plot_k_shot_results(
    results_file: Path,
    metric: Literal["auroc", "tpr_at_fpr", "accuracy"] = "auroc",
    output_file: Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    dataset_name: str | None = None,
) -> None:
    """Plot k-shot fine-tuning results showing performance vs k for different eval_data_usage settings.

    Args:
        results_file: Path to the JSONL file containing k-shot results
        metric: Metric to plot on y-axis. Can be "auroc", "tpr_at_fpr", or "accuracy"
        output_file: Path to save the plot. If None, saves to PLOTS_DIR / "k_shot_results.pdf"
        figsize: Figure size as (width, height) in inches
        dataset_name: Name of dataset to filter results by. If None, uses all datasets.
    """
    # Read results from file
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(KShotResult.model_validate_json(line))

    # Create DataFrame for plotting
    plot_data = []
    k0_metrics = {}  # Store k=0 metrics for each dataset
    for result in results:
        if dataset_name is not None:
            # Extract base dataset name by removing _k{num} suffix
            base_dataset = result.dataset_name.rsplit("_k", 1)[0]
            if not base_dataset.startswith(dataset_name):
                continue
        if result.method == "initial_probe" or result.k == 0:
            k0_metrics[result.dataset_name] = result.metrics[metric]
        else:
            plot_data.append(
                {
                    "k": result.k,
                    "metric": result.metrics[metric],
                    "eval_data_usage": result.config.eval_data_usage,
                    "dataset": result.dataset_name.rsplit("_k", 1)[0],
                }
            )

    df = pd.DataFrame(plot_data)

    if len(df) == 0:
        if dataset_name is not None:
            raise ValueError(f"No results found for dataset '{dataset_name}'")
        else:
            raise ValueError("No results found in the results file")

    # Print number of results per dataset
    print("\nNumber of results per dataset:")
    for dataset in df["dataset"].unique():
        dataset_data = df[df["dataset"] == dataset]
        count = len(dataset_data)
        # Get k value counts
        k_counts = dataset_data["k"].value_counts().sort_index()
        k_str = ", ".join([f"k={k}: {v}" for k, v in k_counts.items()])
        usage_counts = dataset_data["eval_data_usage"].value_counts().to_dict()
        usage_str = ", ".join([f"{k}: {v}" for k, v in usage_counts.items()])
        print(f"{dataset}: {count} results")
        print(f"  k values: {k_str}")
        print(f"  usage: {usage_str}")
    print(f"Total datasets: {len(df['dataset'].unique())}")
    print(f"Total results: {len(df)}")

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot lines for each eval_data_usage setting
    for usage in df["eval_data_usage"].unique():
        usage_data = df[df["eval_data_usage"] == usage]

        if dataset_name is None:
            # When using all datasets, compute mean and std across datasets for each k
            # First group by k and dataset to get mean per dataset
            dataset_means = usage_data.groupby(["k", "dataset"])["metric"].mean()
            # Then group by k to get mean and std of dataset means
            mean_metric = dataset_means.groupby("k").mean()
            std_metric = dataset_means.groupby("k").std()
        else:
            # When using a single dataset, compute mean and std across runs for each k
            mean_metric = usage_data.groupby("k")["metric"].mean()
            std_metric = usage_data.groupby("k")["metric"].std()

        # Get k=0 point
        if usage == "combine":
            # For combine, use the training dataset only baseline
            k0_mean = sum(k0_metrics.values()) / len(k0_metrics)
            k0_std = pd.Series(list(k0_metrics.values())).std()
        else:  # usage == "only"
            # For only, use 0.5 for auroc and 0 for tpr_at_fpr
            k0_mean = 0.5 if metric == "auroc" else 0.0
            k0_std = 0.0  # No variance for these fixed values

        # Plot line with error bars, including k=0 point
        k_0_point = 1
        x_values = [k_0_point] + list(
            mean_metric.index
        )  # Use 0.1 for k=0 to make spacing more consistent
        y_values = [k0_mean] + list(mean_metric.values)
        y_err = [k0_std] + list(std_metric.values)

        plt.errorbar(
            x_values,
            y_values,
            yerr=y_err,
            label=usage,
            marker="o",
            capsize=5,
        )

    # Add horizontal line for k=0 results
    if k0_metrics:
        k0_mean = sum(k0_metrics.values()) / len(k0_metrics)
        k0_std = pd.Series(list(k0_metrics.values())).std()
        plt.axhline(
            y=k0_mean,
            color="gray",
            linestyle="--",
            label="Training data only",
            alpha=0.7,
        )
        # Add error band for k=0
        x_min = k_0_point  # Match the k=0 point position
        x_max = max(df["k"].unique())  # * 1.1  # Add 10% padding
        plt.fill_between(
            [x_min, x_max],
            [k0_mean - k0_std, k0_mean - k0_std],
            [k0_mean + k0_std, k0_mean + k0_std],
            color="gray",
            alpha=0.1,
        )

    # Customize plot
    plt.xlabel("Number of evaluation samples for training")
    plt.ylabel(metric.upper() if metric != "tpr_at_fpr" else "TPR at 1% FPR")
    # plt.title(
    #   "Performance when training on evaluation data"
    #    + (f" - {dataset_name}" if dataset_name else "")
    # )
    plt.legend(title="Training Method")
    plt.grid(True, alpha=0.3)

    # Set x-axis to log scale since k values are powers of 2
    plt.xscale("log", base=2)
    plt.xticks(
        [k_0_point] + list(df["k"].unique()), ["0"] + [str(k) for k in df["k"].unique()]
    )

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if output_file is None:
        output_file = (
            PLOTS_DIR
            / f"k_shot_results_{metric}{f'_{dataset_name}' if dataset_name else ''}.pdf"
        )
    plt.savefig(output_file)
    plt.show()


if __name__ == "__main__":
    # Example usage
    results_file = EVALUATE_PROBES_DIR / "k_shot_fine_tuning_results.jsonl"
    plot_k_shot_results(results_file, metric="auroc")  # , dataset_name="anthropic")
