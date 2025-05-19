import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from models_under_pressure.figures.utils import map_dataset_name

# Set the style
# plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

# Define custom colors for probes
PROBE_COLORS = {
    "Attention": "#FF7F0E",  # HSV: 28°, 95%, 100%
    "Softmax": "#1F77B4",  # HSV: 205°, 83%, 71%
    "Last Token": "#2CA02C",  # Green
    "Max": "#9467BD",  # Purple
    "Mean": "#8C564B",  # Brown
    "Rolling Mean Max": "#E377C2",  # Pink
}

PROBE_NAME_MAPPING = {
    "Rolling Mean Max": "Max of Rolling Means",
}


def process_data(
    paths: list[Path],
    metric_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # First, collect all raw results with their replicate numbers
    results = []
    for path in paths:
        match = re.match(r"results_([a-z]+[a-z_]+)_(dev|test)_(\d)$", path.stem)
        assert match is not None
        probe_name = match.group(1).replace("_", " ").title()
        replicate_num = int(match.group(3))  # Get the replicate number

        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                dataset = map_dataset_name(data.get("dataset_name"))
                metrics = data.get("metrics", {}).get("metrics", {})
                metric_value = metrics.get(metric_name)
                results.append(
                    {
                        "Dataset": dataset,
                        metric_name.upper(): metric_value,
                        "Probe": probe_name,
                        "Replicate": replicate_num,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Calculate mean and standard error for each dataset-probe combination
    stats_df = (
        df.groupby(["Dataset", "Probe"])
        .agg({metric_name.upper(): ["mean", "std", "count"]})
        .reset_index()
    )

    # Flatten column names
    stats_df.columns = [
        "Dataset",
        "Probe",
        f"{metric_name.upper()}_mean",
        f"{metric_name.upper()}_std",
        "count",
    ]

    # Calculate standard error
    stats_df[f"{metric_name.upper()}_se"] = stats_df[
        f"{metric_name.upper()}_std"
    ] / np.sqrt(stats_df["count"])

    # Calculate 95% confidence interval
    # Using t-distribution for small sample sizes
    stats_df[f"{metric_name.upper()}_ci95"] = stats_df[
        f"{metric_name.upper()}_se"
    ] * stats.t.ppf(0.975, stats_df["count"] - 1)

    # Keep only the mean and confidence interval columns
    stats_df = stats_df[
        [
            "Dataset",
            "Probe",
            f"{metric_name.upper()}_mean",
            f"{metric_name.upper()}_ci95",
        ]
    ]

    # Rename columns to match the original format
    stats_df = stats_df.rename(
        columns={
            f"{metric_name.upper()}_mean": metric_name.upper(),
            f"{metric_name.upper()}_ci95": f"{metric_name.upper()}_CI95",
        }
    )

    return stats_df, df  # Return both the processed stats and the raw data


def plot_probe_metric(
    df: tuple[pd.DataFrame, pd.DataFrame],
    metric_name: str,
    output_path: Path,
):
    # Get the raw data from the tuple
    stats_df, raw_df = df

    # Calculate mean AUROC across datasets for each probe and replicate
    replicate_means = (
        raw_df.groupby(["Probe", "Replicate"])[metric_name.upper()].mean().reset_index()
    )

    # Calculate mean and standard error across replicates
    mean_auroc = (
        replicate_means.groupby("Probe")
        .agg({metric_name.upper(): ["mean", "std", "count"]})
        .reset_index()
    )

    # Flatten column names
    mean_auroc.columns = ["Probe", f"{metric_name}_mean", f"{metric_name}_std", "count"]

    # Calculate 95% confidence interval across replicates
    mean_auroc[f"{metric_name}_se"] = mean_auroc[f"{metric_name}_std"] / np.sqrt(
        mean_auroc["count"]
    )
    mean_auroc[f"{metric_name}_ci95"] = mean_auroc[f"{metric_name}_se"] * stats.t.ppf(
        0.975, mean_auroc["count"] - 1
    )

    mean_auroc["dataset"] = "Mean"  # Adding a 'Mean' dataset category

    # Combine the original data with the mean data
    combined_df = pd.concat(
        [
            stats_df,
            mean_auroc[
                ["Probe", "dataset", f"{metric_name}_mean", f"{metric_name}_ci95"]
            ].rename(
                columns={
                    f"{metric_name}_mean": metric_name.upper(),
                    f"{metric_name}_ci95": f"{metric_name.upper()}_CI95",
                }
            ),
        ]
    )

    # Sort probes by their mean AUROC (descending)
    probe_order = mean_auroc.sort_values(f"{metric_name}_mean", ascending=False)[
        "Probe"
    ].tolist()

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
            for d in stats_df["Dataset"].unique()
            if d not in ["Mental Health", "Aya Redteaming"]
        ]
    )
    all_datasets.extend(regular_datasets)

    # Add Mental Health and Aya Redteaming only if they exist in the data
    if "Mental Health" in stats_df["Dataset"].unique():
        all_datasets.append("Mental Health")
    if "Aya Redteaming" in stats_df["Dataset"].unique():
        all_datasets.append("Aya Redteaming")

    combined_df["Dataset"] = pd.Categorical(
        combined_df["Dataset"], categories=all_datasets, ordered=True
    )

    # Sort the dataframe
    combined_df = combined_df.sort_values(["Dataset", "Probe"])

    # Set up the plot
    plt.figure(figsize=(14, 6))

    # Create a mapping from sorted probes to colors
    sorted_probes = overall_means = stats_df.groupby("Probe")[
        metric_name.upper()
    ].mean()
    sorted_probes = sorted_probes.sort_values(ascending=False).index.tolist()

    # Map probe names to colors from our custom palette
    color_mapping = {}
    for probe in sorted_probes:
        if probe in PROBE_COLORS:
            color_mapping[probe] = PROBE_COLORS[probe]
        else:
            # Fallback to a default color if not in our palette
            color_mapping[probe] = "#D3D3D3"  # Light gray as fallback

    # Set up the positions for the bars
    datasets_unique = combined_df["Dataset"].cat.categories.tolist()
    n_datasets = len(datasets_unique)
    n_probes = len(probe_order)
    width = 0.8 / n_probes  # Width of each bar

    # Create positions with extra space after Mean
    group_positions = np.arange(n_datasets, dtype=float)  # Create as float array
    group_positions[1:] += 0.5  # Add extra space after Mean group

    # Calculate overall means for sorting
    overall_means = stats_df.groupby("Probe")[metric_name.upper()].mean()
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

                # Get the metric value and error
                if dataset == "Mean":
                    value = probe_data[f"{metric_name}_mean"].iloc[0]
                    error = probe_data[f"{metric_name}_ci95"].iloc[0]
                else:
                    value = probe_data[metric_name.upper()].iloc[0]
                    error = probe_data[f"{metric_name.upper()}_CI95"].iloc[0]

                # Plot the bar with error bars and black border
                plt.bar(
                    position,
                    value,
                    width=width,
                    label=PROBE_NAME_MAPPING.get(probe, probe)
                    if dataset == datasets_unique[0]
                    else None,  # Only label once
                    color=color_mapping.get(probe, "#D3D3D3"),
                    alpha=0.8
                    if dataset != "Mean"
                    else 1.0,  # Make Mean bars more prominent
                    yerr=error,
                    capsize=3,  # Add caps to error bars
                    edgecolor="black",  # Add black border
                    linewidth=1,  # Set border width
                )

    # Add vertical separator line after Mean
    # plt.axvline(x=0.75, color="gray", linestyle="--", alpha=0.5)

    # Add x-axis labels with Mean in bold using fontweight
    plt.xticks(group_positions, datasets_unique, ha="center", fontsize=14)
    # Make Mean label bold
    for i, label in enumerate(plt.gca().get_xticklabels()):
        if label.get_text() == "Mean":
            label.set_fontweight("bold")

    # plt.legend(
    #     loc="center left",
    #     bbox_to_anchor=(1.02, 0.5),
    #     ncol=1,
    #     framealpha=1.0,  # opaque frame
    #     facecolor="white",
    #     edgecolor="black",
    #     fontsize=14,
    # )

    # Set y-axis limits and labels
    plt.ylim(0.6, 1.0)
    plt.ylabel(metric_name.upper(), fontsize=14)
    plt.yticks(fontsize=14)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save both PNG and PDF versions
    output_path_png = output_path
    output_path_pdf = output_path.with_suffix(".pdf")

    print(f"Saving to {output_path_png} and {output_path_pdf}")
    plt.savefig(output_path_png, bbox_inches="tight", dpi=500)
    plt.savefig(output_path_pdf, bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    from models_under_pressure.config import DATA_DIR

    # Get all files in the evaluate_probes directory
    results_dir = DATA_DIR / "results/evaluate_probes"

    dev_paths = [
        path
        for path in results_dir.glob("*dev*.jsonl")
        if "difference_of_means" not in path.stem
    ]

    test_paths = [
        path
        for path in results_dir.glob("*test*.jsonl")
        if "difference_of_means" not in path.stem
    ]

    # Generate matplotlib version
    plot_probe_metric(
        process_data(dev_paths, "auroc"),
        "AUROC",
        DATA_DIR / "results/plots/probe_auroc_by_dataset_dev.png",
    )

    plot_probe_metric(
        process_data(test_paths, "auroc"),
        "AUROC",
        DATA_DIR / "results/plots/probe_auroc_by_dataset_test.png",
    )
