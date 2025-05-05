import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models_under_pressure.config import EVAL_DATASETS, EVALUATE_PROBES_DIR, RESULTS_DIR
from models_under_pressure.interfaces.dataset import Input, LabelledDataset, to_dialogue


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


def get_dataset_ids_and_token_lengths(
    dataset_paths: list[Path], tokenizer_name: str
) -> pd.DataFrame:
    """
    Returns a datafame where each row is an entry in a dataset.
    """

    print("Loading dataset and calculating token lengths ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data_frames = []

    for dataset_path in dataset_paths:
        # Load dataset
        dataset = LabelledDataset.load_from(dataset_path)

        # Get token lengths for each input
        token_lengths = [len(tokenize(tokenizer, input)) for input in dataset.inputs]

        # Create dataframe
        df = pd.DataFrame(
            {
                "dataset_name": dataset_path.stem,
                "ids": dataset.ids,
                "token_length": token_lengths,
                "label": dataset.labels_numpy(),
            }
        )

        data_frames.append(df)

    # Combine all dataframes
    output = pd.concat(data_frames, ignore_index=True)
    output.set_index(["ids", "dataset_name"], inplace=True)

    return output


def get_probe_results(probe_result_paths: dict[str, Path]) -> pd.DataFrame:
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
    data_frames = []

    for probe_name, probe_result_path in probe_result_paths.items():
        json_data = [json.loads(line) for line in open(probe_result_path)]

        for result in json_data:
            data = {
                "ids": result["ids"],
                "probe_name": probe_name,
                "preds": result["output_labels"],
                "labels": result["ground_truth_labels"],
                "dataset_name": result["dataset_name"],
            }

            df = pd.DataFrame(data)
            df.set_index(["ids", "dataset_name"], inplace=True)

            data_frames.append(df)

    return pd.concat(data_frames, ignore_index=False)


def process_data(
    dataset_paths: list[Path],
    probe_result_paths: dict[str, Path],
    tokenizer_name: str,
    bins: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Load and process the dataset and the probe results. Join the two dataframes on the
    datapoint ids and dataset name. Then group by the probe name, dataset name and token length bins.
    Calculate the AUROC for each probe, dataset and token length bin. Finall calculate the mean AUROC
    over the probe and token length bins. Return this dataframe to be plotted.

    """
    # Get token lengths and ids for each dataset
    df_token_lengths = get_dataset_ids_and_token_lengths(dataset_paths, tokenizer_name)

    # Get probe results
    df_probe_results = get_probe_results(probe_result_paths)

    # Join the dataframes on index (ids and dataset_name)
    df = df_probe_results.join(df_token_lengths, how="inner")

    # Create token length bins
    if bins is None:
        bins = [0, 50, 100, 200, 300, 400, 500, 1000, 2000]

    df["token_length_bin"] = pd.cut(df["token_length"], bins=bins)

    # Group by probe, dataset and token length bin
    grouped = df.groupby(["probe_name", "dataset_name", "token_length_bin"])

    # Calculate AUROC for each group
    results = []
    for name, group in grouped:
        probe_name, dataset_name, length_bin = name

        # Skip if not enough samples
        if len(group) < 2:
            continue

        # Calculate AUROC
        auroc = roc_auc_score(group["labels"], group["preds"])

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


def plot_token_length_bins(df: pd.DataFrame, suffixes: list[str]) -> None:
    """
    For each probe and token length bin, plot the mean AUROC across datasets as a bar chart.
    """
    plt.figure(figsize=(12, 7))

    # Get unique bins and probes
    unique_bins = [str(bin) for bin in df["token_length_bin"].unique()]
    unique_probes = df["probe_name"].unique()

    # Calculate bar positions
    n_bins = len(unique_bins)
    n_probes = len(unique_probes)
    bar_width = 0.8 / n_probes  # Adjust width to fit all bars

    # Plot bars for each probe
    for i, probe_name in enumerate(unique_probes):
        probe_data = df[df["probe_name"] == probe_name]

        # Calculate x positions for this probe's bars
        x_positions = np.arange(n_bins) + (i - (n_probes - 1) / 2) * bar_width

        # Plot bars with error bars
        plt.bar(
            x_positions,
            probe_data["auroc_mean"],
            yerr=probe_data["auroc_std"],
            width=bar_width,
            label=probe_name,
            capsize=5,
            alpha=0.8,
        )

    # Set x-axis ticks and labels
    plt.xticks(range(n_bins), unique_bins, rotation=45, ha="right")

    plt.xlabel("Token Length Bin")
    plt.ylabel("Mean AUROC")
    plt.title("AUROC by Token Length Bin")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7, axis="y")

    # Set y-axis limits to start from 0.5 (typical AUROC range)
    plt.ylim(0.5, 1.0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(RESULTS_DIR / "token_length_bins.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Check the probe results:
    suffixes = [
        "wOgjTcLk_20250505_094202",
        "WYerRCIg_20250505_095716",
    ]
    results_files = {
        suffix: EVALUATE_PROBES_DIR / f"results_{suffix}.jsonl" for suffix in suffixes
    }

    # Create bins for token lengths
    bins = [0, 128, 256, 512, 1024, float("inf")]

    # Process results for each suffix
    final_df = process_data(
        list(EVAL_DATASETS.values()),
        results_files,
        "meta-llama/Llama-3.2-1B-Instruct",
        bins,
    )

    # Plot the results
    plot_token_length_bins(final_df, suffixes)
