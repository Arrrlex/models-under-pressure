#!/usr/bin/env python3
"""
Temporary script to extract AUROC scores for specific models from monitoring cascade results.
"""

import json
import sys
from pathlib import Path

from models_under_pressure.config import LOCAL_MODELS


def extract_auroc_scores(results_dir: str, target_model_names: list[str]):
    """Extract AUROC scores for specified models from cascade results."""

    # Create target models mapping from the list of model names
    target_models = {name: LOCAL_MODELS[name] for name in target_model_names}

    cascade_file = Path(results_dir) / "cascade_results.jsonl"

    if not cascade_file.exists():
        print(f"Error: {cascade_file} not found!")
        return

    # Store results by model and cascade type
    results = {}

    print(f"Reading results from: {cascade_file}")
    print("=" * 60)

    with open(cascade_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            # Only process baseline results with fraction_of_samples = 1.0
            if (
                data.get("cascade_type") in ["baseline", "finetuned_baseline"]
                and data.get("fraction_of_samples") == 1.0
                and data.get("baseline_model_name") in target_models.values()
            ):
                model_name = data["baseline_model_name"]
                cascade_type = data["cascade_type"]
                auroc = data["auroc"]
                dataset = data.get("dataset_name", "unknown")

                # Find friendly name
                friendly_name = None
                for name, full_name in target_models.items():
                    if full_name == model_name:
                        friendly_name = name
                        break

                if friendly_name:
                    if friendly_name not in results:
                        results[friendly_name] = {}
                    if dataset not in results[friendly_name]:
                        results[friendly_name][dataset] = {}
                    results[friendly_name][dataset][cascade_type] = auroc

    # Print results in a nice format
    print("AUROC Scores by Model and Dataset:")
    print("=" * 60)

    for model in target_model_names:
        if model in results:
            print(f"\n{model}:")
            print("-" * 30)
            for dataset, scores in results[model].items():
                print(f"  {dataset}:")
                for cascade_type, auroc in scores.items():
                    type_label = (
                        "Baseline" if cascade_type == "baseline" else "Finetuned"
                    )
                    print(f"    {type_label}: {auroc:.4f}")
        else:
            print(f"\n{model}: No results found")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE - BASELINE AUROC SCORES")
    print("=" * 60)
    print(f"{'Model':<12} {'Dataset':<12} {'Baseline':<10} {'Finetuned':<10}")
    print("-" * 50)

    # Track scores for mean calculation
    model_baseline_scores = {}
    model_finetuned_scores = {}

    for model in target_model_names:
        if model in results:
            model_baseline_scores[model] = []
            model_finetuned_scores[model] = []

            for dataset, scores in results[model].items():
                baseline_score = scores.get("baseline", "N/A")
                finetuned_score = scores.get("finetuned_baseline", "N/A")

                baseline_str = (
                    f"{baseline_score:.4f}" if baseline_score != "N/A" else "N/A"
                )
                finetuned_str = (
                    f"{finetuned_score:.4f}" if finetuned_score != "N/A" else "N/A"
                )

                print(
                    f"{model:<12} {dataset:<12} {baseline_str:<10} {finetuned_str:<10}"
                )

                # Collect scores for mean calculation
                if baseline_score != "N/A":
                    model_baseline_scores[model].append(baseline_score)
                if finetuned_score != "N/A":
                    model_finetuned_scores[model].append(finetuned_score)

    # Calculate and display means
    print("\n" + "=" * 60)
    print("MEAN AUROC SCORES ACROSS ALL DATASETS")
    print("=" * 60)
    print(f"{'Model':<12} {'Baseline Mean':<15} {'Finetuned Mean':<15}")
    print("-" * 50)

    for model in target_model_names:
        if model in model_baseline_scores:
            baseline_mean = (
                sum(model_baseline_scores[model]) / len(model_baseline_scores[model])
                if model_baseline_scores[model]
                else "N/A"
            )
            finetuned_mean = (
                sum(model_finetuned_scores[model]) / len(model_finetuned_scores[model])
                if model_finetuned_scores[model]
                else "N/A"
            )

            baseline_mean_str = (
                f"{baseline_mean:.4f}" if baseline_mean != "N/A" else "N/A"
            )
            finetuned_mean_str = (
                f"{finetuned_mean:.4f}" if finetuned_mean != "N/A" else "N/A"
            )

            print(f"{model:<12} {baseline_mean_str:<15} {finetuned_mean_str:<15}")
        else:
            print(f"{model:<12} {'N/A':<15} {'N/A':<15}")


if __name__ == "__main__":
    # Define target model names (keys from LOCAL_MODELS)
    TARGET_MODEL_NAMES = [
        "llama-1b",
        "llama-8b",
        "gemma-1b",
        "gemma-12b",
        "gemma-27b",
        "llama-70b",
    ]

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to the neurips results directory
        results_dir = "data/results/monitoring_cascade_arxiv"

    extract_auroc_scores(results_dir, TARGET_MODEL_NAMES)
