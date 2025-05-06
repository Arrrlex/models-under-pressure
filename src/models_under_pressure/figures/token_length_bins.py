import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore

from models_under_pressure.config import PROJECT_ROOT
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
    data = []

    for dataset_path in dataset_paths:
        dataset = LabelledDataset.load_from(dataset_path)
        dataset = dataset.assign(
            token_length=[len(tokenize(tokenizer, input)) for input in dataset.inputs]
        )
        data += [
            {
                "dataset_name": dataset_path.stem,
                "id": record.id,
                "token_length": record.token_length,
                "label": record.label.to_int(),
            }
            for record in dataset.to_records()
        ]

    return pd.DataFrame(data).set_index(["id", "dataset_name"])


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
    data = []

    for probe_name, probe_result_path in probe_result_paths.items():
        json_data = [json.loads(line) for line in open(probe_result_path)]

        for result in json_data:
            num_samples = len(result["output_scores"])
            data += [
                {
                    "id": result["ids"][i],
                    "probe_name": probe_name,
                    "preds": result["output_scores"][i],
                    "labels": result["ground_truth_labels"][i],
                    "dataset_name": result["dataset_name"],
                }
                for i in range(num_samples)
            ]

    return pd.DataFrame(data).set_index(["id", "dataset_name"])


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
    df = df_probe_results.join(df_token_lengths, how="inner").reset_index()

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
        # Skip if not enough samples or if we don't have both labels
        if len(group) < 2 or len(group["labels"].unique()) < 2:
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


def plot_token_length_bins(
    df: pd.DataFrame, suffixes: list[str], plot_path: Path
) -> None:
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

        # Add error bands
        # plt.fill_between(
        #     x_positions,
        #     probe_data["auroc_mean"] - probe_data["auroc_std"],
        #     probe_data["auroc_mean"] + probe_data["auroc_std"],
        #     alpha=0.2,
        # )

    # Set x-axis ticks and labels
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha="right")

    plt.xlabel("Token Length Bin")
    plt.ylabel("Mean AUROC")
    plt.title("AUROC by Token Length Bin")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits to start from 0.5 (typical AUROC range)
    plt.ylim(0.5, 1.0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(result_paths: list[Path], bins: list[int], plot_path: Path):
    # Use all available datasets for token length info
    dataset_paths = [
        PROJECT_ROOT / path
        for path in json.loads(open(result_paths[0]).readline())["config"][
            "eval_datasets"
        ]
    ]

    probe_result_paths = {}
    for path in result_paths:
        suffix = path.stem.replace("results_", "")
        model_name = (
            re.search(r"[a-z]+[a-z_]+", suffix)
            .group()
            .removesuffix("_test")
            .replace("_", " ")
            .title()
        )
        probe_result_paths[model_name] = path

    final_df = process_data(
        dataset_paths,
        probe_result_paths,
        "meta-llama/Llama-3.2-1B-Instruct",
        bins,
    )
    plot_token_length_bins(final_df, list(probe_result_paths.keys()), plot_path)


if __name__ == "__main__":
    from models_under_pressure.config import DATA_DIR

    results_dir = DATA_DIR / "results/evaluate_probes"
    result_files = list(results_dir.glob("*.jsonl"))
    test_result_files = [f for f in result_files if "test" in f.stem]
    dev_result_files = [f for f in result_files if f not in test_result_files]

    bins = [0, 128, 256, 512, 1024, float("inf")]

    # Plot for dev
    main(dev_result_files, bins, DATA_DIR / "results/plots/token_length_bins_dev.png")
    # Plot for test
    main(test_result_files, bins, DATA_DIR / "results/plots/token_length_bins_test.png")
