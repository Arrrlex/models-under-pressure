import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from models_under_pressure.config import EVAL_DATASETS, EVALUATE_PROBES_DIR


def plot_metric_from_results(
    suffixes: list[str],
    metric_name: str,
    data_dir: str = str(EVALUATE_PROBES_DIR),
    datasets: list[str] | None = None,
    title_fontsize: int = 16,
    label_fontsize: int = 12,
    save_path: str | None = None,
):
    """
    Plots a styled bar chart of the specified metric for each model (suffix) across datasets.

    Parameters:
        suffixes (list of str): List of suffixes (e.g., ["attention_light", "baseline"])
        metric_name (str): Metric to extract and plot (e.g., "auroc")
        data_dir (str): Path to the folder containing result_*.jsonl files
        datasets (list of str, optional): Fixed list of dataset names to plot in order
        title_fontsize (int): Font size for the plot title
        label_fontsize (int): Font size for axis labels and tick labels

    Returns:
        pd.DataFrame: The pivoted DataFrame used for plotting
    """
    results = []

    for suffix in suffixes:
        model_name = (
            re.search(r"[a-z]+[a-z_]+", suffix)
            .group()
            .removesuffix("_test")
            .replace("_", " ")
            .title()
        )
        file_path = os.path.join(data_dir, f"results_{suffix}.jsonl")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                dataset = data.get("dataset_name")
                metric_value = data.get("metrics").get("metrics").get(metric_name)
                if dataset and metric_value is not None:
                    results.append(
                        {
                            "Dataset": dataset,
                            "Metric": metric_value,
                            "Model": model_name,
                        }
                    )

    df = pd.DataFrame(results)

    def map_dataset_name(dataset: str) -> str:
        if "manual" in dataset:
            return "Manual"
        elif "anthropic" in dataset:
            return "Anthropic"
        elif "toolace" in dataset:
            return "Toolace"
        elif "mts" in dataset:
            return "MTS"
        elif "mt" in dataset:
            return "MT"
        elif "mask" in dataset:
            return "Mask"
        elif "mental_health" in dataset:
            return "Mental Health"
        elif "aya" in dataset or "redteaming" in dataset:
            return "Aya Redteaming"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    df["Dataset"] = df["Dataset"].map(map_dataset_name)

    # Calculate mean AUROC for each model and find the best one
    mean_aurocs = df.groupby("Model")["Metric"].mean().sort_values(ascending=False)
    best_model = mean_aurocs.index[0]
    second_best_model = mean_aurocs.index[1]
    print(
        f"\nBest model by mean AUROC: {best_model} (mean AUROC: {mean_aurocs[best_model]:.3f})"
    )
    print(
        f"Second best model by mean AUROC: {second_best_model} (mean AUROC: {mean_aurocs[second_best_model]:.3f})"
    )

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), width_ratios=[1, 2])

    # Plot mean metrics in the first subplot
    mean_aurocs.plot(
        kind="bar", ax=ax1, width=0.8, edgecolor="black", alpha=0.9, color="lightblue"
    )
    ax1.set_title(
        f"Mean {metric_name.upper()}\nAcross All Datasets", fontsize=title_fontsize
    )
    ax1.set_ylabel(f"Mean {metric_name.upper()}", fontsize=label_fontsize)
    ax1.tick_params(axis="x", labelrotation=30, labelsize=label_fontsize)
    ax1.tick_params(axis="y", labelsize=label_fontsize)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of each bar
    for i, v in enumerate(mean_aurocs):
        ax1.text(
            i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=label_fontsize - 2
        )

    # Pivot the DataFrame to get datasets as index and models as columns
    pivot_df = df.pivot(index="Dataset", columns="Model", values="Metric")
    # Reorder columns to match the sorted mean values
    pivot_df = pivot_df[mean_aurocs.index]

    # Create the bar plot with side-by-side bars in the second subplot
    pivot_df.plot(kind="bar", ax=ax2, width=0.8, edgecolor="black", alpha=0.9)

    ax2.set_ylabel(metric_name.upper(), fontsize=label_fontsize)
    ax2.set_xlabel("Dataset", fontsize=label_fontsize)
    ax2.set_title(f"{metric_name.upper()} Across Datasets", fontsize=title_fontsize)
    ax2.tick_params(axis="x", labelrotation=30, labelsize=label_fontsize)
    ax2.tick_params(axis="y", labelsize=label_fontsize)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.legend(
        title="Model",
        fontsize=label_fontsize - 1,
        title_fontsize=label_fontsize,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.savefig(f"{metric_name}_across_datasets.png", bbox_inches="tight")

    return df


if __name__ == "__main__":
    from models_under_pressure.config import DATA_DIR

    # Get all files in the evaluate_probes directory
    results_dir = DATA_DIR / "results/evaluate_probes"
    result_files = list(results_dir.glob("*.jsonl"))
    test_result_files = [f for f in result_files if "test" in f.stem]
    dev_result_files = [f for f in result_files if f not in test_result_files]

    # Extract unique suffixes by removing common prefix/extension
    dev_suffixes = sorted(
        {file.stem.replace("results_", "") for file in dev_result_files}
    )
    test_suffixes = sorted(
        {file.stem.replace("results_", "") for file in test_result_files}
    )

    print(f"Found dev suffixes: {dev_suffixes}")
    print(f"Found test suffixes: {test_suffixes}")
    plot_metric_from_results(
        suffixes=dev_suffixes,
        metric_name="auroc",
        datasets=list(EVAL_DATASETS.keys()),
        title_fontsize=18,
        label_fontsize=12,
        save_path=str(DATA_DIR / "results/plots/auroc_dev.png"),
    )
    plot_metric_from_results(
        suffixes=test_suffixes,
        metric_name="auroc",
        datasets=list(EVAL_DATASETS.keys()),
        title_fontsize=18,
        label_fontsize=12,
        save_path=str(DATA_DIR / "results/plots/auroc_test.png"),
    )
