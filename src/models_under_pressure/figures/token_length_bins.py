import json
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from models_under_pressure.config import EVAL_DATASETS, EVALUATE_PROBES_DIR, RESULTS_DIR


def load_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".jsonl":
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        return pd.DataFrame(data)
    elif file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def prepare_dataset(
    files: list[Path],
    result_suffixes: list[str],
    tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    bins: list[int] | None = None,
) -> pd.DataFrame:
    if bins is None:
        bins = [0, 50, 100, 200, 300, 400, 500, 1000, 2000]

    # Load tokenizer (replace with llama-70b tokenizer as needed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data_frames = []

    for file_path, suffix in product(files, result_suffixes):
        df = load_file(file_path)
        result_file = os.path.join(EVALUATE_PROBES_DIR, f"results_{suffix}.jsonl")
        results = []
        with open(result_file, "r") as f:
            for line in f:
                results.append(json.loads(line))

        # Assuming results are in the same order, add prediction
        predictions = [
            r["output_labels"] for r in results if r["dataset_name"] == file_path.stem
        ][0]
        df["prediction"] = predictions

        # Rename ground truth column
        if "labels" in df.columns:
            df["ground_truth"] = (df["labels"] == "high-stakes").astype(int)
        # Tokenize inputs to get length
        df["input_length"] = df["inputs"].apply(
            lambda x: len(tokenizer.encode(x, add_special_tokens=False))
        )
        df["suffix"] = suffix

        data_frames.append(df)

    full_df = pd.concat(data_frames, ignore_index=True)

    # Bin by input length
    full_df["length_bin"] = pd.cut(full_df["input_length"], bins=bins)

    return full_df


# Group and sort
def plot_stakes_accuracy(
    full_df: pd.DataFrame,
    suffixes: list[str],
    stakes_type: str = "high-stakes",
) -> None:
    grouped = full_df.groupby("length_bin")
    bin_labels = []
    totals = {suffix: [] for suffix in suffixes}
    corrects = {suffix: [] for suffix in suffixes}
    lower_ranges = []

    for bin_range, group in grouped:
        bin_labels.append(str(bin_range))
        lower_ranges.append(float(bin_range.left))  # type: ignore
        if stakes_type == "all":
            stakes = group
        else:
            stakes = group[group["ground_truth"] == stakes_type]

        for suffix in suffixes:
            suffix_data = stakes[stakes["suffix"] == suffix]
            totals[suffix].append(len(suffix_data))
            corrects[suffix].append(
                roc_auc_score(suffix_data["ground_truth"], suffix_data["prediction"])
            )
        # sort lower ranges and accordingly sort bin labels and corrects[suffix]
    sorted_idx = np.argsort(lower_ranges)
    bin_labels = [bin_labels[i] for i in sorted_idx]
    corrects = {
        suffix: [corrects[suffix][i] for i in sorted_idx] for suffix in suffixes
    }

    x = np.arange(len(bin_labels))  # label locations

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#001524",
        "#15616D",
        "#058C42",
        "#FF7D00",
        "#78290F",
        "#A53860",
        "#E63946",
        "#9D4EDD",
        "#1DD3B0",
        "#6D597A",
        "#433E3F",
    ]

    # Plot bars for each suffix
    # Create a DataFrame for seaborn plotting
    plot_data = []
    for i, suffix in enumerate(suffixes):
        for bin_idx, bin_label in enumerate(bin_labels):
            plot_data.append(
                {
                    "Input Length Bin": bin_label,
                    "AUROC": corrects[suffix][bin_idx],
                    "Method": suffix,
                }
            )
    plot_df = pd.DataFrame(plot_data)

    # Create seaborn bar plot
    sns.barplot(
        data=plot_df,
        x="Input Length Bin",
        y="AUROC",
        hue="Method",
        ax=ax,
        palette=colors,
    )

    ax.set_xlabel("Input Length Bins")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC w.r.t Input Length")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_ylim(0.4, 1.0)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{stakes_type}_token_length_bins.png")


# Example usage:
suffixes = [
    "attention_light",
    "attention_then_linear",
    "per_token_mean",
    "sklearn_mean",
    "difference_of_means",
    "per_token_max-of-rolling-mean_40",
    "per_token_max-of-sentence-means",
    "per_token_mean_of_top_10",
    "per_token_max",
    "per_token_last",
    "per_entry",
]
full_df = prepare_dataset(
    list(EVAL_DATASETS.values()),
    suffixes,
    tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
)
full_df.to_csv(RESULTS_DIR / "stakes_token_length_bins.csv", index=False)
# full_df = pd.read_csv(RESULTS_DIR / "stakes_token_length_bins.csv")
plot_stakes_accuracy(full_df, suffixes, stakes_type="all")
