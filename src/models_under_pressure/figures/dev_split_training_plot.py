import json
from pathlib import Path
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models_under_pressure.config import EVALUATE_PROBES_DIR, PLOTS_DIR, RESULTS_DIR
from models_under_pressure.interfaces.results import DevSplitResult

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


def load_baseline_results(baseline_file: Path) -> Dict[str, float]:
    """Load baseline results from a file.

    Args:
        baseline_file: Path to the JSONL file containing baseline results

    Returns:
        Dictionary mapping dataset names to baseline AUROC values
    """
    baseline_results = {}
    with open(baseline_file) as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                dataset_name = result.get("dataset_name")
                if dataset_name:
                    baseline_results[dataset_name] = result.get("auroc", 0.0)
                else:
                    print(
                        f"WARNING: No dataset name found in baseline results for {line}"
                    )

    return baseline_results


def plot_dev_split_results(
    results_file: Path,
    metric: Literal["auroc", "tpr_at_fpr", "accuracy"] = "auroc",
    output_file: Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    dataset_name: str | None = None,
    combine_datasets: bool = True,
    baseline_file: Optional[Path] = None,
) -> None:
    """Plot k-shot fine-tuning results showing performance vs k for different dev_sample_usage settings.

    Args:
        results_file: Path to the JSONL file containing k-shot results
        metric: Metric to plot on y-axis. Can be "auroc", "tpr_at_fpr", or "accuracy"
        output_file: Path to save the plot. If None, saves to PLOTS_DIR / "k_shot_results.pdf"
        figsize: Figure size as (width, height) in inches
        dataset_name: Name of dataset to filter results by. If None, uses all datasets.
        combine_datasets: If True, combines all datasets into a single line per dev_sample_usage.
                        If False, plots individual lines for each dataset with consistent colors
                        and line styles.
        baseline_file: Path to a JSONL file containing baseline results to plot as a horizontal line
    """
    # Read results from file
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(DevSplitResult.model_validate_json(line))

    # Load baseline results if provided
    baseline_results = {}
    if baseline_file is not None and baseline_file.exists():
        baseline_results = load_baseline_results(baseline_file)

    eval_usage_mapping = {
        "only": "Dev samples only",
        "combine": "Synthetic + dev samples",
    }

    # Create DataFrame for plotting
    plot_data = []
    k0_metrics = {}  # Store k=0 metrics for each dataset
    for result in results:
        if dataset_name is not None:
            # Extract base dataset name by removing _k{num} suffix
            base_dataset = result.dataset_name.rsplit("_k", 1)[0]
            if base_dataset != dataset_name:
                continue
        if result.method == "initial_probe" or result.k == 0:
            k0_metrics[result.dataset_name] = result.metrics[metric]
        else:
            plot_data.append(
                {
                    "k": result.k,
                    "metric": result.metrics[metric],
                    "dev_sample_usage": eval_usage_mapping[
                        result.config.dev_sample_usage
                    ],
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
        usage_counts = dataset_data["dev_sample_usage"].value_counts().to_dict()
        usage_str = ", ".join([f"{k}: {v}" for k, v in usage_counts.items()])
        print(f"{dataset}: {count} results")
        print(f"  k values: {k_str}")
        print(f"  usage: {usage_str}")
    print(f"Total datasets: {len(df['dataset'].unique())}")
    print(f"Total results: {len(df)}")

    # Create the plot
    plt.figure(figsize=figsize)

    # Define line styles for different dev_sample_usage settings
    line_styles = {
        eval_usage_mapping["only"]: "-",
        eval_usage_mapping["combine"]: "--",
    }

    if combine_datasets:
        # Plot lines for each dev_sample_usage setting
        for usage in df["dev_sample_usage"].unique():
            usage_data = df[df["dev_sample_usage"] == usage]

            if dataset_name is None:
                # For each k, align runs by order for each dataset, then compute mean/std over run means (mean across datasets for each run)
                mean_metric = {}
                std_metric = {}
                for k in sorted(usage_data["k"].unique()):
                    k_data = usage_data[usage_data["k"] == k]
                    # Get list of datasets
                    datasets = sorted(k_data["dataset"].unique())
                    # For each dataset, get the list of runs (in order)
                    run_lists = []
                    for dataset in datasets:
                        runs = k_data[k_data["dataset"] == dataset]["metric"].values
                        run_lists.append(runs)
                    # Stack into 2D array (datasets x runs), then transpose to (runs x datasets)
                    run_matrix = np.array([runs for runs in run_lists])
                    # Only keep columns (runs) where all datasets have a value (align by shortest)
                    min_runs = min(len(runs) for runs in run_lists)
                    if min_runs == 0:
                        continue  # skip if no runs for this k
                    run_matrix = run_matrix[:, :min_runs]
                    # Now, for each run index, compute mean across datasets
                    run_means = run_matrix.mean(axis=0)  # shape: (min_runs,)
                    # Compute mean and std over these run means
                    mean_metric[k] = run_means.mean()
                    std_metric[k] = run_means.std()
                mean_metric = pd.Series(mean_metric)
                std_metric = pd.Series(std_metric)
            else:
                # When using a single dataset, compute mean and std across runs for each k
                mean_metric = usage_data.groupby("k")["metric"].mean()
                std_metric = usage_data.groupby("k")["metric"].std()

            # Get k=0 point
            if usage == eval_usage_mapping["combine"]:
                # For combine, use the training dataset only baseline
                # Align runs by order across datasets for k=0
                k0_run_lists = []
                for dataset in sorted(k0_metrics.keys()):
                    val = k0_metrics[dataset]
                    if isinstance(val, (list, np.ndarray)):
                        runs = np.array(val)
                    else:
                        runs = np.array([val])
                    k0_run_lists.append(runs)
                min_k0_runs = min(len(runs) for runs in k0_run_lists)
                if min_k0_runs > 0:
                    k0_matrix = np.array([runs[:min_k0_runs] for runs in k0_run_lists])
                    k0_run_means = k0_matrix.mean(axis=0)
                    k0_mean = k0_run_means.mean()
                    k0_std = k0_run_means.std()
            elif usage == eval_usage_mapping["only"]:  # usage == "only"
                # For only, use 0.5 for auroc and 0 for tpr_at_fpr
                k0_mean = 0.5 if metric == "auroc" else 0.0
                k0_std = 0.0  # No variance for these fixed values
            else:
                raise NotImplementedError(f"Didn't implement dev_sample_usage: {usage}")

            # Plot line with error bars, including k=0 point
            k_0_point = 1
            x_values = [k_0_point] + list(mean_metric.index)
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

            # Print performance values for this usage
            print(f"\nPerformance values for {usage}:")
            print(f"  k=0: {k0_mean:.4f} ± {k0_std:.4f}")
            for k, mean, std in zip(
                mean_metric.index, mean_metric.values, std_metric.values
            ):
                print(f"  k={k}: {mean:.4f} ± {std:.4f}")
    else:
        # Plot individual lines for each dataset and dev_sample_usage combination
        # Get unique datasets and assign colors
        datasets = sorted(df["dataset"].unique())
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(datasets)))
        dataset_colors = dict(zip(datasets, colors))

        for dataset in datasets:
            dataset_data = df[df["dataset"] == dataset]
            # Plot both strategies for this dataset
            for usage in eval_usage_mapping.values():
                usage_data = dataset_data[dataset_data["dev_sample_usage"] == usage]
                if len(usage_data) == 0:
                    continue  # Skip if no data for this usage type

                # Compute mean and std across runs for each k
                mean_metric = usage_data.groupby("k")["metric"].mean()
                std_metric = usage_data.groupby("k")["metric"].std()

                # Get k=0 point
                if usage == eval_usage_mapping["combine"]:
                    # For combine, use the training dataset only baseline
                    val = k0_metrics.get(dataset)
                    if val is not None:
                        if isinstance(val, (list, np.ndarray)):
                            k0_mean = np.mean(val)
                            k0_std = np.std(val)
                        else:
                            k0_mean = val
                            k0_std = 0.0
                        # Include k=0 point
                        k_0_point = 1
                        x_values = np.array([k_0_point] + list(mean_metric.index))
                        y_values = np.array([k0_mean] + list(mean_metric.values))
                        y_err = np.array([k0_std] + list(std_metric.values))
                    else:
                        # Skip k=0 point but still plot the rest
                        x_values = np.array(list(mean_metric.index))
                        y_values = np.array(list(mean_metric.values))
                        y_err = np.array(list(std_metric.values))
                elif usage == eval_usage_mapping["only"]:
                    # For only, use 0.5 for auroc and 0 for tpr_at_fpr
                    k0_mean = 0.5 if metric == "auroc" else 0.0
                    k0_std = 0.0
                    k_0_point = 1
                    x_values = np.array([k_0_point] + list(mean_metric.index))
                    y_values = np.array([k0_mean] + list(mean_metric.values))
                    y_err = np.array([k0_std] + list(std_metric.values))
                else:
                    raise NotImplementedError(
                        f"Didn't implement dev_sample_usage: {usage}"
                    )

                plt.errorbar(
                    x_values,
                    y_values,
                    yerr=y_err,
                    label=f"{dataset} ({usage})",
                    marker="o",
                    capsize=5,
                    color=dataset_colors[dataset],
                    linestyle=line_styles[usage],
                )

                # Print performance values for this dataset and usage
                print(f"\nPerformance values for {dataset} ({usage}):")
                if "k0_mean" in locals():
                    print(f"  k=0: {k0_mean:.4f} ± {k0_std:.4f}")
                for k, mean, std in zip(
                    mean_metric.index, mean_metric.values, std_metric.values
                ):
                    print(f"  k={k}: {mean:.4f} ± {std:.4f}")

    # Add horizontal line for k=0 results
    if k0_metrics and combine_datasets:
        # Align runs by order across datasets for k=0
        k0_run_lists = []
        for dataset in sorted(k0_metrics.keys()):
            val = k0_metrics[dataset]
            if isinstance(val, (list, np.ndarray)):
                runs = np.array(val)
            else:
                runs = np.array([val])
            k0_run_lists.append(runs)
        min_k0_runs = min(len(runs) for runs in k0_run_lists)
        if min_k0_runs > 0:
            k0_matrix = np.array([runs[:min_k0_runs] for runs in k0_run_lists])
            k0_run_means = k0_matrix.mean(axis=0)
            k0_mean = k0_run_means.mean()
            k0_std = k0_run_means.std()
            plt.axhline(
                y=k0_mean,
                color="gray",
                linestyle="--",
                label="Synthetic dataset only",
                alpha=0.7,
            )
            # Add error band for k=0
            x_min = 0  # Match the k=0 point position
            x_max = max(df["k"].unique())
            plt.fill_between(
                [x_min, x_max],
                [k0_mean - k0_std, k0_mean - k0_std],
                [k0_mean + k0_std, k0_mean + k0_std],
                color="gray",
                alpha=0.1,
            )

            # Print synthetic dataset only performance
            print(f"\nSynthetic dataset only performance: {k0_mean:.4f} ± {k0_std:.4f}")

    # Add baseline results if provided
    if baseline_results:
        current_dataset = dataset_name if dataset_name else "default"
        if current_dataset in baseline_results:
            baseline_value = float(baseline_results[current_dataset])
            plt.axhline(
                y=baseline_value,
                color="red",
                linestyle="-.",
                label="Baseline results",
                alpha=0.7,
            )
            print(f"\nBaseline result from file: {baseline_value:.4f}")
        else:
            # If we have baseline results but not for this specific dataset,
            # compute the mean across all datasets
            mean_baseline = float(np.mean(list(baseline_results.values())))
            plt.axhline(
                y=mean_baseline,
                color="red",
                linestyle="-.",
                label="Baseline results (mean)",
                alpha=0.7,
            )
            print(
                f"\nBaseline result from file (mean across datasets): {mean_baseline:.4f}"
            )

    # Customize plot
    plt.xlabel("Number of dev split samples for training")
    plt.ylabel(
        "Mean " + (metric.upper() if metric != "tpr_at_fpr" else "TPR at 1% FPR")
    )
    plt.legend(title="Training Data", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Set x-axis to log scale since k values are powers of 2
    plt.xscale("log", base=2)
    plt.xticks(
        [k_0_point] + list(df["k"].unique()), ["0"] + [str(k) for k in df["k"].unique()]
    )

    # Adjust layout to accommodate legend
    plt.tight_layout()

    # Save plot
    if output_file is None:
        output_file = (
            PLOTS_DIR
            / f"dev_split_training_results_{metric}{f'_{dataset_name}' if dataset_name else ''}.pdf"
        )
    plt.savefig(output_file)
    # plt.show()


if __name__ == "__main__":
    results_file = EVALUATE_PROBES_DIR / "dev_split_training_test.jsonl"
    # Optional: path to baseline results file
    baseline_file = (
        RESULTS_DIR / "baseline_llama-70b.jsonl"
    )  # Uncomment and modify to use
    plot_dev_split_results(
        results_file,
        metric="auroc",
        combine_datasets=False,
        baseline_file=baseline_file,  # Uncomment to use baseline results
    )
