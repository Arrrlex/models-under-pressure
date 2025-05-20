import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import RESULTS_DIR
from models_under_pressure.figures.utils import get_baseline_results, get_probe_results


def process_raw_probe_results(result: dict) -> dict:
    """
    Flatten the metrics dictionary in the result.
    """
    metrics = result["metrics"]
    for metric_name, metric_value in metrics.items():
        result[metric_name] = metric_value

    # Extract the dataset name from the probe_spec

    return result


def get_data_efficiency_probe_results(file_paths: list[Path]) -> pd.DataFrame:
    """
    For each file in the file_paths variable, check the path exists and raise an error if it doesn't.
    Load in the results from each file and concat the dataframes together.
    Return the concatenated dataframe.
    """

    print(f"Loading data efficiency probe results from {len(file_paths)} files...")

    all_results = []

    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    result = json.loads(line)
                    # Flatten the metrics dictionary

                    breakpoint()
                    if "metrics" in result:
                        metrics = result["metrics"]
                        for metric_name, metric_value in metrics.items():
                            result[metric_name] = metric_value

                    all_results.append(result)

    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} data efficiency probe results")
    return df


def prepare_plot_df(
    df_probes: list[pd.DataFrame],
    df_baselines: list[pd.DataFrame],
    min_train_size: int = 0,
) -> pd.DataFrame:
    """
    Combine the baseline and probe results into a single plot dataframe.

    Args:
        df_probes: List of probe dataframes
        df_baselines: List of baseline dataframes
        min_train_size: Minimum training dataset size to include (default: 0)
    """

    # Concat the probe dataframes:
    df_probe = pd.concat(df_probes)
    df_baseline = pd.concat(df_baselines)

    # Rename the probe output_scores to scores:
    df_probe = df_probe.rename(columns={"output_scores": "scores"})
    df_probe["method"] = df_probe["probe_name"]

    df_baseline = df_baseline.rename(columns={"ground_truth": "ground_truth_labels"})
    # Remove the model maker name (anything before '/') from the model_name
    df_baseline["model_name_short"] = df_baseline["model_name"].apply(
        lambda x: x.split("/")[-1] if "/" in x else x
    )
    df_baseline["method"] = "finetuned_" + df_baseline["model_name_short"]

    selected_columns = [
        "scores",
        "ground_truth_labels",
        "method",
        "dataset_name",
        "train_dataset_size",
    ]

    df = pd.concat([df_probe[selected_columns], df_baseline[selected_columns]])

    # Filter out points below min_train_size
    df = df[df["train_dataset_size"] >= min_train_size]

    grp = (
        df.groupby(["dataset_name", "method", "train_dataset_size"])
        .apply(lambda x: float(roc_auc_score(x["ground_truth_labels"], x["scores"])))
        .reset_index()
        .rename(columns={0: "auroc"})
    )

    # Take the mean across the dataset_name column:
    grp = grp.groupby(["method", "train_dataset_size"]).agg(
        auroc=("auroc", "mean"),
    )

    return grp


def plot_results(df: pd.DataFrame) -> None:
    """
    Plot a line chart where the x-axis is the train_dataset_size and the y-axis is the auroc for each method.
    """

    # Reset the index to make train_dataset_size and method columns
    plot_df = df.reset_index()

    # Set up the plot style
    plt.figure(figsize=(7, 6))
    sns.set_style("whitegrid")

    # Get a list of distinct colors from a colormap
    # unique_methods = plot_df["method"].unique()
    # num_methods = len(unique_methods)
    # color_palette = cm.get_cmap("tab10", num_methods)
    method_to_color = {
        "attention": "#FF7F0E",
        "linear_then_softmax": "#1F77B4",
        "sklearn": "brown",
        "finetuned_Llama-3.1-8B-Instruct": "#008000",
        "finetuned_Llama-3.2-1B-Instruct": "#7CFC00",
    }

    # Define markers for methods
    markers = ["o", "s", "^", "D", "v", ">", "<", "p", "*", "h"]

    # Plot each method as a separate line
    for i, method in enumerate(plot_df["method"].unique()):
        print(f"Plotting {method}")
        method_df = plot_df[plot_df["method"] == method]
        plt.plot(
            method_df["train_dataset_size"],
            method_df["auroc"],
            label=method,
            marker=markers[i % len(markers)],
            color=method_to_color[method],
            linewidth=2,
            markersize=8,
        )

    # Set x-axis to log scale
    plt.xscale("log")

    # Add labels and title
    plt.xlabel("Training Dataset Size (samples)", fontsize=14)
    plt.ylabel("AUROC (avg across datasets)", fontsize=14)
    # plt.title("Probe Performance vs Training Data Size", fontsize=16)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.legend(fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "data_efficiency.png", dpi=600, bbox_inches="tight")
    plt.savefig(output_dir / "data_efficiency.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # Define the directory containing the results
    results_dir = RESULTS_DIR / "data_efficiency"

    # Find all JSONL files in the directory
    all_jsonl_files = list(results_dir.glob("*.jsonl"))

    # Separate files into probe and baseline paths based on filename
    probe_paths = [path for path in all_jsonl_files if "_baseline_" not in path.name]
    baseline_paths = [path for path in all_jsonl_files if "_baseline_" in path.name]

    print(
        f"Found {len(probe_paths)} probe files and {len(baseline_paths)} baseline files"
    )

    df_probe = get_probe_results(probe_paths)
    df_baseline = get_baseline_results(baseline_paths)

    # Set minimum training size to 1000 samples
    min_train_size = 4
    df_plot = prepare_plot_df([df_probe], [df_baseline], min_train_size=min_train_size)
    plot_results(df_plot)
