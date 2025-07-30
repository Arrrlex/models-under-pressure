#!/usr/bin/env python3
"""
Script to calculate mean FLOPs per sample for Gemma-27B on test datasets
using the attention probe calculation (2 * activation_dim * token_count).
"""

# Import from the main project
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models_under_pressure.config import LOCAL_MODELS, PROJECT_ROOT
from models_under_pressure.interfaces.dataset import LabelledDataset, to_dialogue


def get_activation_dim(full_model_name: str) -> int:
    model_name = next(k for k, v in LOCAL_MODELS.items() if v == full_model_name)
    if model_name == "llama-1b":
        activation_dim = 2048
    elif model_name == "gemma-1b":
        activation_dim = 1152
    elif model_name == "llama-8b":
        activation_dim = 4096
    elif model_name == "llama-70b":
        activation_dim = 8192
    elif model_name == "gemma-12b":
        activation_dim = 3840
    elif model_name == "gemma-27b":
        activation_dim = 5376
    else:
        raise ValueError(f"Unknown activation dimension for model: {model_name}")
    return int(activation_dim)


def tokenize_input(tokenizer: AutoTokenizer, input_data) -> int:
    """Tokenize a single input and return the token count."""
    dialogue = to_dialogue(input_data)
    input_dicts = [[d.model_dump() for d in dialogue]]

    input_str = tokenizer.apply_chat_template(
        input_dicts,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokens = tokenizer(input_str, return_tensors="pt")
    # Count tokens using attention mask (excluding the first token as per codebase pattern)
    token_count = tokens["attention_mask"][:, 1:].sum().item()
    return token_count


def calculate_probe_flops(
    probe_type: str, token_count: int, activation_dim: int
) -> int:
    """Calculate FLOPs for attention probe: 2 * activation_dim * token_count."""
    if probe_type == "attention":
        return 2 * activation_dim * token_count
    else:
        return activation_dim * token_count


def load_test_datasets() -> Dict[str, Path]:
    """Load test dataset paths from config."""
    config_path = PROJECT_ROOT / "config" / "eval_datasets" / "test_balanced_fixed.yaml"
    with open(config_path) as f:
        datasets = yaml.safe_load(f)

    # Convert relative paths to absolute paths
    return {name: PROJECT_ROOT / path for name, path in datasets.items()}


def main(probe_type: str, model_name: str):
    # Model configuration for Gemma-27B
    model_name = LOCAL_MODELS[model_name]
    activation_dim = get_activation_dim(model_name)

    print(f"Model: {model_name}")
    print(f"Activation dimension: {activation_dim}")
    print(f"FLOP calculation: 2 * {activation_dim} * token_count")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load test datasets
    test_datasets = load_test_datasets()
    print(f"Found {len(test_datasets)} test datasets:")
    for name, path in test_datasets.items():
        print(f"  - {name}: {path}")
    print()

    all_token_counts = []
    all_flops = []
    results_by_dataset = {}

    # Process each dataset
    for dataset_name, dataset_path in test_datasets.items():
        print(f"Processing {dataset_name}...")

        # Load dataset
        dataset = LabelledDataset.load_from(dataset_path)
        print(f"  Loaded {len(dataset)} samples")

        # Tokenize all inputs and calculate token counts
        token_counts = []
        for i, input_data in enumerate(dataset.inputs):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(dataset.inputs)} samples...")

            token_count = tokenize_input(tokenizer, input_data)
            token_counts.append(token_count)

        # Calculate FLOPs for each sample
        flops_per_sample = [
            calculate_probe_flops(probe_type, count, activation_dim)
            for count in token_counts
        ]

        # Calculate statistics
        mean_tokens = np.mean(token_counts)
        mean_flops = np.mean(flops_per_sample)

        results_by_dataset[dataset_name] = {
            "num_samples": len(dataset),
            "mean_tokens": mean_tokens,
            "mean_flops": mean_flops,
            "min_tokens": np.min(token_counts),
            "max_tokens": np.max(token_counts),
            "min_flops": np.min(flops_per_sample),
            "max_flops": np.max(flops_per_sample),
        }

        # Add to overall statistics
        all_token_counts.extend(token_counts)
        all_flops.extend(flops_per_sample)

        print(f"  Mean tokens per sample: {mean_tokens:.1f}")
        print(f"  Mean FLOPs per sample: {mean_flops:.0f}")
        print()

    # Overall statistics
    overall_mean_tokens = np.mean(all_token_counts)
    overall_mean_flops = np.mean(all_flops)

    print("=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Activation dimension: {activation_dim}")
    print(f"Total samples across all datasets: {len(all_token_counts)}")
    print()

    print("Per-dataset results:")
    for dataset_name, results in results_by_dataset.items():
        print(
            f"{dataset_name:15} | {results['num_samples']:4d} samples | "
            f"{results['mean_tokens']:6.1f} tokens | "
            f"{results['mean_flops']:10.0f} FLOPs"
        )

    print("-" * 60)
    print(
        f"{'OVERALL':15} | {len(all_token_counts):4d} samples | "
        f"{overall_mean_tokens:6.1f} tokens | "
        f"{overall_mean_flops:10.0f} FLOPs"
    )
    print("=" * 60)

    print(
        f"\nMean FLOPs per sample for Gemma-27B attention probe: {overall_mean_flops:.0f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
