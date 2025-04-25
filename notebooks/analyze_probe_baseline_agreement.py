import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models_under_pressure.config import EVAL_DATASETS
from models_under_pressure.dataset_utils import load_dataset


def load_results(file_path: Path) -> Dict:
    """Load results from a JSONL file."""
    with open(file_path) as f:
        return json.loads(f.read())


def create_agreement_matrix(
    probe_results: Dict, baseline_results: Dict
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Create a matrix showing agreement between probe and baseline predictions.

    Returns:
        Tuple of (agreement matrix DataFrame, dictionary mapping categories to IDs)
    """
    # Extract predictions and ground truth
    probe_predictions = np.array(probe_results["output_labels"])
    baseline_predictions = np.array(baseline_results["labels"])
    ground_truth = np.array(baseline_results["ground_truth"])
    ids = baseline_results["ids"]

    # Create agreement matrix
    matrix = np.zeros((2, 2), dtype=int)

    # Create dictionary to store IDs for each category
    category_ids = {
        "both_correct": [],
        "probe_correct_baseline_wrong": [],
        "probe_wrong_baseline_correct": [],
        "both_wrong": [],
    }

    # Fill matrix and collect IDs
    for i in range(len(ground_truth)):
        probe_correct = probe_predictions[i] == ground_truth[i]
        baseline_correct = baseline_predictions[i] == ground_truth[i]

        if probe_correct and baseline_correct:
            matrix[0, 0] += 1
            category_ids["both_correct"].append(ids[i])
        elif probe_correct and not baseline_correct:
            matrix[0, 1] += 1
            category_ids["probe_correct_baseline_wrong"].append(ids[i])
        elif not probe_correct and baseline_correct:
            matrix[1, 0] += 1
            category_ids["probe_wrong_baseline_correct"].append(ids[i])
        else:
            matrix[1, 1] += 1
            category_ids["both_wrong"].append(ids[i])

    # Convert to DataFrame
    df = pd.DataFrame(
        matrix,
        index=["Probe Correct", "Probe Wrong"],
        columns=["Baseline Correct", "Baseline Wrong"],
    )

    return df, category_ids


def sample_cases(
    probe_results: Dict,
    baseline_results: Dict,
    category_ids: Dict[str, List[str]],
    dataset_name: str,
    n_samples: int = 10,
) -> pd.DataFrame:
    """Sample cases from each category and create a DataFrame with the results."""
    # Load the dataset to get the input texts
    dataset = load_dataset(
        dataset_path=EVAL_DATASETS[dataset_name],
        model_name=baseline_results["model_name"],
        layer=None,  # We don't need activations
        compute_activations=False,
    )

    # Create a mapping from ID to index in the dataset
    id_to_idx = {id_: idx for idx, id_ in enumerate(dataset.ids)}

    samples = []

    for category, ids in category_ids.items():
        if len(ids) == 0:
            continue

        # Sample IDs
        sampled_ids = random.sample(ids, min(n_samples, len(ids)))

        for id_ in sampled_ids:
            # Get the index in the dataset for this ID
            dataset_idx = id_to_idx[id_]

            # Find the index in the results for this ID
            results_idx = baseline_results["ids"].index(id_)

            sample = {
                "category": category,
                "id": id_,
                "text": dataset.inputs[dataset_idx],
                "ground_truth": baseline_results["ground_truth"][results_idx],
                "probe_prediction": probe_results["output_labels"][results_idx],
                "baseline_prediction": baseline_results["labels"][results_idx],
                "probe_score": probe_results["output_scores"][results_idx],
                "baseline_score": baseline_results["high_stakes_scores"][results_idx],
            }
            samples.append(sample)

    return pd.DataFrame(samples)


def main():
    # Paths to results files
    results_dir = Path("data/results/monitoring_cascade")
    probe_results_path = results_dir / "probe_results.jsonl"
    baseline_results_path = results_dir / "baseline_llama-70b.jsonl"

    # Load results
    probe_results = load_results(probe_results_path)
    baseline_results = load_results(baseline_results_path)

    # Create agreement matrix and get category IDs
    agreement_matrix, category_ids = create_agreement_matrix(
        probe_results, baseline_results
    )

    # Print results
    print("\nAgreement Matrix between Probe and Llama-70b Baseline:")
    print(agreement_matrix)

    # Calculate and print percentages
    total_samples = agreement_matrix.sum().sum()
    print("\nPercentages:")
    print((agreement_matrix / total_samples * 100).round(1))

    # Print some statistics
    print("\nStatistics:")
    print(f"Total samples: {total_samples}")
    print(
        f"Probe accuracy: {(agreement_matrix.iloc[0].sum() / total_samples * 100):.1f}%"
    )
    print(
        f"Baseline accuracy: {(agreement_matrix.iloc[:, 0].sum() / total_samples * 100):.1f}%"
    )
    print(
        f"Agreement rate: {((agreement_matrix.iloc[0, 0] + agreement_matrix.iloc[1, 1]) / total_samples * 100):.1f}%"
    )

    # Sample cases and save to CSV
    samples_df = sample_cases(
        probe_results,
        baseline_results,
        category_ids,
        dataset_name=baseline_results["dataset_name"],
        n_samples=20,
    )
    file_name = "probe_baseline_agreement_samples.csv"
    samples_df.to_csv(results_dir / file_name, index=False)
    print(f"Saved samples to {results_dir / file_name}")


if __name__ == "__main__":
    main()
