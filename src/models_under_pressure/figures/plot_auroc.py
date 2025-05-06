import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def process_data(
    paths: list[Path],
    metric_name: str,
) -> pd.DataFrame:
    results = []
    for path in paths:
        probe_name = (
            re.search(r"[a-z]+[a-z_]+$", path.stem)
            .group()
            .removesuffix("_test")
            .replace("_", " ")
            .title()
        )
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                dataset = map_dataset_name(data.get("dataset_name"))
                metric_value = data.get("metrics").get("metrics").get(metric_name)
                results.append(
                    {
                        "Dataset": dataset,
                        metric_name.upper(): metric_value,
                        "Probe": probe_name,
                    }
                )

    return pd.DataFrame(results)


def plot_probe_metric(
    df: pd.DataFrame,
    metric_name: str,
    output_path: Path,
):
    # Calculate mean for each probe across all datasets
    mean_auroc = df.groupby("Probe")[metric_name].mean().reset_index()
    mean_auroc["dataset"] = "Mean"  # Adding a 'Mean' dataset category

    # Combine the original data with the mean data
    combined_df = pd.concat([df, mean_auroc])

    # Sort probes by their mean AUROC (descending)
    probe_order = mean_auroc.sort_values(metric_name, ascending=False)["Probe"].tolist()

    # Create a categorical type with the ordered probes
    combined_df["Probe"] = pd.Categorical(
        combined_df["Probe"], categories=probe_order, ordered=True
    )

    # Create custom dataset order with 'Mean' as the first group
    all_datasets = ["Mean"]
    # Add all regular datasets first
    regular_datasets = sorted(
        [
            d
            for d in df["Dataset"].unique()
            if d not in ["Mental Health", "Aya Redteaming"]
        ]
    )
    all_datasets.extend(regular_datasets)

    # Add Mental Health and Aya Redteaming only if they exist in the data
    if "Mental Health" in df["Dataset"].unique():
        all_datasets.append("Mental Health")
    if "Aya Redteaming" in df["Dataset"].unique():
        all_datasets.append("Aya Redteaming")

    combined_df["Dataset"] = pd.Categorical(
        combined_df["Dataset"], categories=all_datasets, ordered=True
    )

    # Sort the dataframe
    combined_df = combined_df.sort_values(["Dataset", "Probe"])

    # Set up the plot
    plt.figure(figsize=(14, 8))

    # Create a custom color palette
    colors = sns.color_palette("muted", len(probe_order))
    probe_colors = dict(zip(probe_order, colors))

    # Set up the positions for the bars
    datasets_unique = combined_df["Dataset"].cat.categories.tolist()
    n_datasets = len(datasets_unique)
    n_probes = len(probe_order)
    width = 0.8 / n_probes  # Width of each bar

    # Create positions with extra space after Mean
    group_positions = np.arange(n_datasets, dtype=float)  # Create as float array
    group_positions[1:] += 0.5  # Add extra space after Mean group

    # Calculate overall means for sorting
    overall_means = df.groupby("Probe")[metric_name].mean()
    sorted_probes = overall_means.sort_values(ascending=False).index.tolist()

    # Plot each dataset group
    for dataset in datasets_unique:
        if dataset == "Mean":
            # Use the pre-calculated means for the Mean group
            dataset_data = mean_auroc
        else:
            dataset_data = combined_df[combined_df["Dataset"] == dataset]

        # Plot each probe as a set of bars
        for i, probe in enumerate(sorted_probes):
            probe_data = dataset_data[dataset_data["Probe"] == probe]
            if not probe_data.empty:
                # Calculate position for this probe's bar
                position = group_positions[datasets_unique.index(dataset)] + width * (
                    i - n_probes / 2 + 0.5
                )

                # Plot the bar
                plt.bar(
                    position,
                    probe_data[metric_name].iloc[0],
                    width=width,
                    label=probe
                    if dataset == datasets_unique[0]
                    else None,  # Only label once
                    color=probe_colors[probe],
                    alpha=0.8
                    if dataset != "Mean"
                    else 1.0,  # Make Mean bars more prominent
                )

    # Add vertical separator line after Mean
    plt.axvline(x=0.75, color="gray", linestyle="--", alpha=0.5)

    # Add x-axis labels with Mean in bold using fontweight
    plt.xticks(group_positions, datasets_unique, ha="center")
    # Make Mean label bold
    for i, label in enumerate(plt.gca().get_xticklabels()):
        if label.get_text() == "Mean":
            label.set_fontweight("bold")

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set y-axis limits
    plt.ylim(0.5, 1.0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    from models_under_pressure.config import DATA_DIR

    # Get all files in the evaluate_probes directory
    results_dir = DATA_DIR / "results/evaluate_probes"
    result_files = list(results_dir.glob("*.jsonl"))
    test_result_files = [f for f in result_files if "test" in f.stem]
    dev_result_files = [f for f in result_files if f not in test_result_files]

    df = process_data(dev_result_files, "auroc")
    plot_probe_metric(
        df, "AUROC", DATA_DIR / "results/plots/probe_auroc_by_dataset_dev.png"
    )

    df = process_data(test_result_files, "auroc")
    plot_probe_metric(
        df,
        "AUROC",
        DATA_DIR / "results/plots/probe_auroc_by_dataset_test.png",
    )
