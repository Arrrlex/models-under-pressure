import json
import os

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
                        {"Dataset": dataset, "Metric": metric_value, "Model": suffix}
                    )

    df = pd.DataFrame(results)
    data_dict = {
        "manual_upsampled": "Manual",
        "anthropic_balanced_apr_23": "Anthropic",
        "toolace_balanced_apr_22": "Toolace",
        "mt_balanced_apr_16": "MT",
        "mts_balanced_apr_22": "MTS",
        "mask_samples_balanced": "Mask",
    }

    # Map dataset names using data_dict
    df["Dataset"] = df["Dataset"].map(data_dict).fillna(df["Dataset"])

    # Calculate mean AUROC for each model and find the best one
    mean_aurocs = df.groupby("Model")["Metric"].mean()
    best_model = mean_aurocs.idxmax()
    second_best_model = mean_aurocs.nlargest(2).index[1]
    print(
        f"\nBest model by mean AUROC: {best_model} (mean AUROC: {mean_aurocs[best_model]:.3f})"
    )
    print(
        f"Second best model by mean AUROC: {second_best_model} (mean AUROC: {mean_aurocs[second_best_model]:.3f})"
    )

    # Pivot the DataFrame to get datasets as index and models as columns
    pivot_df = df.pivot(index="Dataset", columns="Model", values="Metric")

    # Create the bar plot with side-by-side bars
    ax = pivot_df.plot(
        kind="bar", figsize=(12, 7), width=0.8, edgecolor="black", alpha=0.9
    )

    ax.set_ylabel(metric_name.upper(), fontsize=label_fontsize)
    ax.set_xlabel("Dataset", fontsize=label_fontsize)
    ax.set_title(f"{metric_name.upper()} Across Datasets", fontsize=title_fontsize)
    ax.tick_params(axis="x", labelrotation=30, labelsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.legend(
        title="Model",
        fontsize=label_fontsize - 1,
        title_fontsize=label_fontsize,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(f"{metric_name}_across_datasets.png", bbox_inches="tight")

    return df


plot_metric_from_results(
    suffixes=[
        "attention_light_50_val_loss",
        "attention_then_linear",
        "per_token_mean",
        "sklearn_mean",
    ],
    metric_name="auroc",
    datasets=list(EVAL_DATASETS.keys()),
    title_fontsize=18,
    label_fontsize=12,
)
