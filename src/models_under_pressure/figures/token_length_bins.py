import json
from pathlib import Path
from typing import Optional

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

    print("Loading dataset and calculating token lengths")
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
    data_frames = []

    for probe_name, probe_result_path in probe_result_paths.items():
        json_data = [json.loads(line) for line in open(probe_result_path)]

        for result in json_data:
            data = {
                "ids": result["ids"],
                "probe_name": probe_name,
                "preds": result["output_scores"],
                "labels": result["output_labels"],
                "dataset_name": result["dataset_name"],
            }

            df = pd.DataFrame(data)
            df.set_index(["ids", "dataset_name"], inplace=True)

            data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)


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
    For each probe and token length bin, plot the mean AUROC across datasets.
    """

    # Set up the plot style
    plt.style.use("seaborn")
    plt.figure(figsize=(10, 6))

    # Plot each probe's results
    for probe_name in df["probe_name"].unique():
        probe_data = df[df["probe_name"] == probe_name]

        # Plot mean AUROC with error bars
        plt.errorbar(
            probe_data["token_length_bin"],
            probe_data["auroc_mean"],
            yerr=probe_data["auroc_std"],
            label=probe_name,
            marker="o",
            capsize=5,
        )

    plt.xlabel("Token Length Bin")
    plt.ylabel("Mean AUROC")
    plt.title("AUROC by Token Length Bin")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(RESULTS_DIR / "token_length_bins.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Check the probe results:
    suffixes = ["sklearn_prompts_4x", "sklearn_prompts_2x"]
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
