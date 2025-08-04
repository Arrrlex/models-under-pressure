#!/usr/bin/env python3
"""
Script to analyze probe results and identify interesting samples based on prediction scores
and ground truth stakes labels. Prints the actual input text for the selected samples.

Usage:
    python analyze_results.py <results_file_path> <dataset_name> [--k <number_of_samples>]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from models_under_pressure.interfaces.dataset import LabelledDataset


def load_results(file_path: str, dataset_name: str) -> Dict[str, Any]:
    """Load results from JSONL file for the specified dataset."""
    try:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                if data.get("dataset_name").startswith(dataset_name + "_"):
                    print(f"Loading {data.get('dataset_name')}")
                    return data

        print(f"Error: Dataset '{dataset_name}' not found in results file")
        sys.exit(1)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        sys.exit(1)


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load dataset using the existing LabelledDataset infrastructure and create a mapping from IDs to inputs."""
    try:
        dataset = LabelledDataset.from_jsonl(Path(dataset_path))

        # Create mapping from IDs to inputs
        id_to_input = {}
        for i, (input_item, id_item) in enumerate(zip(dataset.inputs, dataset.ids)):
            id_to_input[id_item] = input_item

        return id_to_input
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset '{dataset_path}': {e}")
        sys.exit(1)


def print_overall_performance(data: Dict[str, Any]) -> None:
    """Print overall performance metrics."""
    print("=" * 60)
    print(f"OVERALL PERFORMANCE - {data['dataset_name']}")
    print("=" * 60)

    metrics = data.get("metrics", {}).get("metrics", {})
    if not metrics:
        print("No metrics available")
        return

    print(f"Method: {data.get('method', 'Unknown')}")
    print(f"AUROC: {metrics.get('auroc', 'N/A'):.4f}")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"TPR at FPR: {metrics.get('tpr_at_fpr', 'N/A'):.4f}")
    print(f"FPR: {metrics.get('fpr', 'N/A'):.4f}")

    if "best_epoch" in data:
        print(f"Best Epoch: {data['best_epoch']}")

    print()


def get_interesting_samples(
    output_scores: List[float],
    ground_truth_labels: List[int],
    ground_truth_scale_labels: List[int],
    ids: List[str],
    k: int,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Find k samples with highest scores but low-stakes ground truth labels
    and k samples with lowest scores but high-stakes ground truth labels.

    Returns:
        Tuple of (high_score_low_stakes, low_score_high_stakes) lists
    """
    # Combine all data for easier processing
    combined_data = list(
        zip(output_scores, ground_truth_labels, ground_truth_scale_labels, ids)
    )

    # Filter for low-stakes samples (assuming lower scale values mean low stakes)
    # and high-stakes samples (assuming higher scale values mean high stakes)
    low_stakes_samples = [
        sample for sample in combined_data if sample[2] <= 3
    ]  # Low stakes threshold
    high_stakes_samples = [
        sample for sample in combined_data if sample[2] >= 8
    ]  # High stakes threshold

    # Sort by prediction scores
    # High scores with low stakes (potential false positives in high-stakes contexts)
    high_score_low_stakes = sorted(
        low_stakes_samples, key=lambda x: x[0], reverse=True
    )[:k]

    # Low scores with high stakes (potential false negatives in high-stakes contexts)
    low_score_high_stakes = sorted(high_stakes_samples, key=lambda x: x[0])[:k]

    return high_score_low_stakes, low_score_high_stakes


def print_samples(
    samples: List[Tuple], title: str, id_to_input: Dict[str, Any]
) -> None:
    """Print a list of samples with their actual inputs."""
    print("-" * 80)
    print(title)
    print("-" * 80)

    if not samples:
        print("No samples found matching criteria")
        return

    for i, (score, gt_label, stakes, sample_id) in enumerate(samples, 1):
        print(f"\n{i}. Sample ID: {sample_id}")
        print(f"   Score: {score:.4f} | Ground Truth: {gt_label} | Stakes: {stakes}")
        print("   Input:")

        # Get the input (now it's either a string or a Dialogue object)
        input_item = id_to_input.get(sample_id, "Input not found")
        if input_item != "Input not found":
            if isinstance(input_item, str):
                # If it's a string, print as is
                print(f"      USER: {input_item}")
            elif isinstance(input_item, list):
                # If it's a dialogue (list of messages), print each message
                for message in input_item:
                    if hasattr(message, "role") and hasattr(message, "content"):
                        # It's a Message object
                        print(f"      {message.role.upper()}: {message.content}")
                    elif isinstance(message, dict):
                        # It's a dictionary
                        role = message.get("role", "unknown")
                        content = message.get("content", "")
                        print(f"      {role.upper()}: {content}")
                    else:
                        print(f"      {message}")
            else:
                print(f"      {input_item}")
        else:
            print(f"      {input_item}")
        print("-" * 40)

    print()


def main(results_file: str, dataset_name: str, k: int):
    # Load results
    data = load_results(results_file, dataset_name)

    # Print overall performance
    print_overall_performance(data)

    # Determine dataset path based on dataset name
    dataset_path = data.get("dataset_path")
    if not dataset_path:
        print("Error: Dataset path not found in results data")
        sys.exit(1)

    # Load dataset to get input mapping
    print(f"Loading dataset from: {dataset_path}")
    id_to_input = load_dataset(dataset_path)

    # Extract required fields
    try:
        output_scores = data["output_scores"]
        ground_truth_labels = data["ground_truth_labels"]
        ground_truth_scale_labels = data["ground_truth_scale_labels"]
        ids = data["ids"]
    except KeyError as e:
        print(f"Error: Missing required field in results: {e}")
        sys.exit(1)

    # Validate data consistency
    if not all(
        len(arr) == len(output_scores)
        for arr in [ground_truth_labels, ground_truth_scale_labels, ids]
    ):
        print("Error: Inconsistent array lengths in results data")
        sys.exit(1)

    print(f"Total samples: {len(output_scores)}")
    print()

    # Get interesting samples
    high_score_low_stakes, low_score_high_stakes = get_interesting_samples(
        output_scores, ground_truth_labels, ground_truth_scale_labels, ids, k
    )

    # Print results
    print_samples(
        high_score_low_stakes,
        f"TOP {k} SAMPLES: High Prediction Scores + Low-Stakes Ground Truth",
        id_to_input,
    )

    print_samples(
        low_score_high_stakes,
        f"TOP {k} SAMPLES: Low Prediction Scores + High-Stakes Ground Truth",
        id_to_input,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze probe results and identify interesting samples"
    )
    parser.add_argument("results_file", help="Path to the results JSONL file")
    parser.add_argument("dataset_name", help="Name of the dataset to analyze")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of samples to show for each category (default: 10)",
    )

    args = parser.parse_args()
    main(args.results_file, args.dataset_name, args.k)
