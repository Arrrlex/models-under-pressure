import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore

from models_under_pressure.config import DATA_DIR, PROJECT_ROOT
from models_under_pressure.interfaces.dataset import (
    Input,
    LabelledDataset,
    to_dialogue,
)


def tokenize(tokenizer: PreTrainedTokenizerBase, input: Input) -> torch.Tensor:
    dialogue = to_dialogue(input)
    input_dicts = [[d.model_dump() for d in dialogue]]

    input_str = tokenizer.apply_chat_template(
        input_dicts,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokens = tokenizer(input_str)  # type: ignore

    return tokens["input_ids"][0]  # type: ignore


def get_dataset_name(path_or_name: Path | str) -> str:
    dataset_name = path_or_name.stem if isinstance(path_or_name, Path) else path_or_name
    if "manual" in dataset_name:
        return "Manual"
    elif "anthropic" in dataset_name:
        return "Anthropic"
    elif "toolace" in dataset_name:
        return "Toolace"
    elif "mts" in dataset_name:
        return "MTS"
    elif "mt" in dataset_name:
        return "MT"
    elif "mask" in dataset_name:
        return "Mask"
    elif "mental_health" in dataset_name:
        return "Mental Health"
    elif "aya" in dataset_name or "redteaming" in dataset_name:
        return "Aya Redteaming"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_ids_and_token_lengths(
    dataset_paths: list[Path], tokenizer_name: str
) -> pd.DataFrame:
    """
    Returns a datafame where each row is an entry in a dataset.
    """

    print("Loading dataset and calculating token lengths ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = []

    for dataset_path in dataset_paths:
        dataset = LabelledDataset.load_from(dataset_path)
        dataset = dataset.assign(
            token_length=[len(tokenize(tokenizer, input)) for input in dataset.inputs]
        )
        data += [
            {
                "dataset_name": get_dataset_name(dataset_path),
                "id": record.id,
                "token_length": record.token_length,
            }
            for record in dataset.to_records()
        ]

    return pd.DataFrame(data)


def get_probe_results(probe_result_paths: list[Path]) -> pd.DataFrame:
    """
    Read the results file, return a dataframe where each row is an entry in a dataset.
    The dataset is structured as:
    - probe_name -> probe from which the results are generated
    - preds -> list of predictions for each input TODO: logits or probit?
    - dataset_name -> the name of the dataset from which the results are generated
    - labels -> list of labels for each inputs
    - ids -> ids for each datapoint within it's dataset

    """
    print("Loading probe results ... ")
    data = []

    for probe_result_path in probe_result_paths:
        json_data = [json.loads(line) for line in open(probe_result_path)]

        probe_name = (
            json_data[0]["config"]["probe_spec"]["name"].replace("_", " ").title()
        )

        for result in json_data:
            num_samples = len(result["output_scores"])
            data += [
                {
                    "id": result["ids"][i],
                    "probe_name": probe_name,
                    "pred": result["output_scores"][i],
                    "label": result["ground_truth_labels"][i],
                    "dataset_name": get_dataset_name(result["dataset_name"]),
                }
                for i in range(num_samples)
            ]

    return pd.DataFrame(data)


def get_probe_results_with_token_lengths(
    probe_result_paths: list[Path], tokenizer_name: str
) -> pd.DataFrame:
    """
    Read the results file, return a dataframe where each row is an entry in a dataset.
    Only includes data points that have results for all probes.
    """
    dataset_paths = set()
    for i, probe_result_path in enumerate(probe_result_paths):
        _dataset_paths = json.loads(open(probe_result_path).readline())["config"][
            "eval_datasets"
        ]
        dataset_paths = {PROJECT_ROOT / path for path in _dataset_paths}
        if i == 0:
            all_dataset_paths = dataset_paths
        else:
            assert dataset_paths == all_dataset_paths

    probe_scores = get_probe_results(probe_result_paths)
    token_lengths = get_dataset_ids_and_token_lengths(
        list(dataset_paths), tokenizer_name
    )

    # Get the set of (id, dataset_name) pairs that have results for all probes
    probe_counts = probe_scores.groupby(["id", "dataset_name"]).size()
    complete_points = probe_counts[probe_counts == len(probe_result_paths)].index

    # Filter both dataframes to only include complete points
    probe_scores = (
        probe_scores.set_index(["id", "dataset_name"])
        .loc[complete_points]
        .reset_index()
    )
    token_lengths = (
        token_lengths.set_index(["id", "dataset_name"])
        .loc[complete_points]
        .reset_index()
    )

    return probe_scores.merge(token_lengths, how="inner", on=["id", "dataset_name"])


def process_data(
    df: pd.DataFrame,
    bins: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Load and process the dataset and the probe results. Join the two dataframes on the
    datapoint ids and dataset name. Then group by the probe name, dataset name and token length bins.
    Calculate the AUROC for each probe, dataset and token length bin. Finall calculate the mean AUROC
    over the probe and token length bins. Return this dataframe to be plotted.

    """

    df = df.copy()

    # Create token length bins
    if bins is None:
        bins = [0, 50, 100, 200, 300, 400, 500, 1000, 2000]

    df["token_length_bin"] = pd.cut(df["token_length"], bins=bins)

    # Group by probe, dataset and token length bin
    grouped = df.groupby(
        ["probe_name", "dataset_name", "token_length_bin"], observed=True
    )

    # Calculate AUROC for each group
    results = []
    for name, group in grouped:
        probe_name, dataset_name, length_bin = name
        # Skip if not enough samples or if we don't have both labels
        if len(group) < 2 or len(group["label"].unique()) < 2:
            continue

        # Calculate AUROC
        auroc = roc_auc_score(group["label"], group["pred"])

        results.append(
            {
                "probe_name": probe_name,
                "dataset_name": dataset_name,
                "token_length_bin": length_bin,
                "auroc": auroc,
                "num_samples": len(group),
            }
        )

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Calculate mean AUROC across datasets for each probe and bin
    final_df = (
        results_df.groupby(["probe_name", "token_length_bin"])
        .agg({"auroc": ["mean", "std", "count"], "num_samples": "sum"})
        .reset_index()
    )

    # Flatten column names
    final_df.columns = [
        "probe_name",
        "token_length_bin",
        "auroc_mean",
        "auroc_std",
        "num_datasets",
        "total_samples",
    ]

    return final_df


def plot_token_length_bins(df: pd.DataFrame, plot_path: Path) -> None:
    """
    For each probe, plot the mean AUROC across datasets as a line plot with error bands.
    """
    plt.figure(figsize=(12, 7))

    # Get unique probes
    unique_probes = df["probe_name"].unique()

    # Plot lines for each probe
    for probe_name in unique_probes:
        probe_data = df[df["probe_name"] == probe_name].sort_values("token_length_bin")

        # Extract bin midpoints for x-axis
        bin_labels = [str(bin) for bin in probe_data["token_length_bin"]]
        x_positions = range(len(bin_labels))

        # Plot line with error bands
        plt.plot(
            x_positions,
            probe_data["auroc_mean"],
            label=probe_name,
            marker="o",
            markersize=6,
            linewidth=2,
        )

    # Set x-axis ticks and labels
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha="right")

    plt.xlabel("Token Length Bin")
    plt.ylabel("Mean AUROC")
    plt.title("AUROC by Token Length Bin")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits based on data
    y_min = df["auroc_mean"].min() - 0.05
    y_max = df["auroc_mean"].max() + 0.05
    plt.ylim(y_min, y_max)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rolling_auroc(df: pd.DataFrame, plot_path: Path, window_size: int) -> None:
    """
    Plot rolling AUROC over token length by taking rolling windows of IDs and calculating
    AUROC for each probe within those windows.

    Args:
        df: DataFrame containing token lengths, predictions, and labels
        plot_path: Path to save the plot
        window_size: Number of IDs to include in each rolling window
    """
    plt.figure(figsize=(12, 7))

    # Get unique IDs and sort them by token length
    unique_ids = df[["id", "dataset_name", "token_length"]].drop_duplicates()
    unique_ids = unique_ids.sort_values("token_length")

    # Get unique probes
    unique_probes = df["probe_name"].unique()

    # Store all AUROC values to determine y-axis limits
    all_aurocs = []

    # Calculate rolling AUROC for each probe
    for probe_name in unique_probes:
        # Filter data for this probe
        probe_data = df[df["probe_name"] == probe_name]

        # Calculate rolling AUROC
        rolling_auroc = []
        token_lengths = []

        stride = window_size // 25

        # Take rolling windows of IDs
        for i in tqdm(
            list(range(0, len(unique_ids) - window_size + 1, stride)),
            desc=f"Calculating rolling AUROC for {probe_name}",
        ):
            # Get the IDs in this window
            window_ids = unique_ids.iloc[i : i + window_size]

            # Get all data points for these IDs for this probe
            window_data = probe_data.merge(
                window_ids[["id", "dataset_name"]], on=["id", "dataset_name"]
            )

            if len(window_data["label"].unique()) < 2:
                continue

            auroc = roc_auc_score(window_data["label"], window_data["pred"])
            rolling_auroc.append(auroc)
            token_lengths.append(window_ids["token_length"].mean())

        all_aurocs.extend(rolling_auroc)

        # Plot line
        plt.plot(
            token_lengths,
            rolling_auroc,
            label=probe_name,
            linewidth=2,
        )

    plt.xlabel("Token Length")
    plt.ylabel("Rolling AUROC")
    plt.title(f"Rolling AUROC (Window Size: {window_size} IDs)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits based on data
    y_min = min(all_aurocs) - 0.05
    y_max = max(all_aurocs) + 0.05
    plt.ylim(y_min, y_max)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_token_length_distributions(df: pd.DataFrame, plot_path: Path) -> None:
    # Plot histograms of token lengths by dataset, one per row
    datasets = sorted(df["dataset_name"].unique())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(n_datasets, 1, figsize=(10, 3 * n_datasets))
    fig.suptitle("Token Length Distributions by Dataset", y=0.95)

    for idx, dataset in enumerate(datasets):
        dataset_df = df[df["dataset_name"] == dataset]
        # Take just one probe's data since token length is the same for all probes
        dataset_df = dataset_df[
            dataset_df["probe_name"] == dataset_df["probe_name"].iloc[0]
        ]

        axes[idx].hist(dataset_df["token_length"], bins=30, alpha=0.7, color=f"C{idx}")
        axes[idx].set_title(dataset)
        axes[idx].set_xlabel("Token Length")
        axes[idx].set_ylabel("Count")

    plt.tight_layout()

    # Save the token length distribution plot
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(
    result_paths: list[Path],
    bins: list[int],
    suffix: str,
    probes: list[str],
    window_size: int,
):
    plots_dir = DATA_DIR / "results/plots"
    # Use all available datasets for token length info
    df = get_probe_results_with_token_lengths(
        result_paths, "meta-llama/Llama-3.2-1B-Instruct"
    )

    # Filter for selected probes
    df = df[df["probe_name"].isin(probes)]

    bins_df = process_data(df, bins)
    plot_token_length_bins(bins_df, plots_dir / f"auroc_token_length_bins_{suffix}.png")

    # Create rolling average plot
    rolling_plot_path = plots_dir / f"auroc_token_length_rolling_{suffix}.png"
    plot_rolling_auroc(df, rolling_plot_path, window_size=window_size)

    plot_token_length_distributions(
        df, plots_dir / f"token_length_distributions_{suffix}.png"
    )


if __name__ == "__main__":
    results_dir = DATA_DIR / "results/evaluate_probes"
    test_result_files = set(results_dir.glob("*test*.jsonl"))
    dev_result_files = set(results_dir.glob("*.jsonl")) - test_result_files

    bins = [0, 128, 256, 512, 1024, float("inf")]

    selected_probes = ["Attention", "Linear Then Mean", "Linear Then Softmax"]

    # Plot for dev
    main(
        list(dev_result_files),
        bins,
        suffix="dev",
        probes=selected_probes,
        window_size=500,
    )
    # Plot for test
    main(
        list(test_result_files),
        bins,
        suffix="test",
        probes=selected_probes,
        window_size=500,
    )
