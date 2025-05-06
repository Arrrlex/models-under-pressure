from pathlib import Path

import pandas as pd

from models_under_pressure.figures.utils import (
    get_baseline_results,
    get_continuation_results,
    get_probe_results,
)


def prepare_data(
    probe_paths: list[Path],
    baseline_paths: list[Path],
    continuation_paths: list[Path],
) -> pd.DataFrame:
    probe_results = get_probe_results(probe_paths)
    baseline_results = get_baseline_results(baseline_paths)
    continuation_results = get_continuation_results(continuation_paths)

    """
    Load the probe and baseline results, combine the dataframes on the ids and dataset_name
    ensuring the probe and baseline results have different column names
    
    """

    # Rename columns output_scores, and output_labels to scores and labels
    probe_results.rename(
        columns={"output_scores": "probe_scores", "output_labels": "probe_labels"},
        inplace=True,
    )
    baseline_results.rename(
        columns={
            "scores": "baseline_scores",
        },
        inplace=True,
    )
    continuation_results.rename(
        columns={
            "high_stakes_scores": "continuation_scores",
        },
        inplace=True,
    )

    # Add prefix to probe columns
    probe_cols = [
        col
        for col in probe_results.columns
        if col
        not in [
            "probe_scores",
            "probe_labels",
            "probe_name",
            "probe_spec",
            "dataset_name",
            "ids",
        ]
    ]
    probe_results.rename(
        columns={col: f"probe_{col}" for col in probe_cols}, inplace=True
    )

    # Add prefix to baseline columns
    baseline_cols = [
        col
        for col in baseline_results.columns
        if col not in ["baseline_scores", "dataset_name", "ids"]
    ]
    baseline_results.rename(
        columns={col: f"baseline_{col}" for col in baseline_cols}, inplace=True
    )

    # Add prefix to continuation columns
    continuation_cols = [
        col
        for col in continuation_results.columns
        if col not in ["continuation_scores", "dataset_name", "ids"]
    ]
    continuation_results.rename(
        columns={col: f"continuation_{col}" for col in continuation_cols}, inplace=True
    )

    # Combine the results on the ids and dataset_name columns
    df_combined_results = pd.merge(
        probe_results, baseline_results, on=["ids", "dataset_name"], how="inner"
    )

    df_combined_results = pd.merge(
        df_combined_results,
        continuation_results,
        on=["ids", "dataset_name"],
        how="inner",
    )

    return df_combined_results


if __name__ == "__main__":
    baseline_path = "/home/ubuntu/models-under-pressure/data/results/finetune_baselines_Llama-3.1-8B-Instruct__0_2025-05-06.jsonl"
    probe_path = "/home/ubuntu/models-under-pressure/data/results/evaluate_probes/results_wOgjTcLk_20250505_094202.jsonl"
    contin_path = (
        "/home/ubuntu/models-under-pressure/data/results/baseline_llama-70b.jsonl"
    )

    df_combined = prepare_data(
        probe_paths=[Path(probe_path)],
        baseline_paths=[Path(baseline_path)],
        continuation_paths=[Path(contin_path)],
    )

    breakpoint()
