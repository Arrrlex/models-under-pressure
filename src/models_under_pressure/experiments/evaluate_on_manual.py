# Script to train linear probes on training dataset and evaluate on manual eval dataset
import argparse
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    MANUAL_DATASET_PATH,
    SYNTHETIC_DATASET_PATH,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.experiments.train_probes import train_probes_and_save_results
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import ProbeEvaluationResults


def load_manual_eval_dataset(
    manual_dataset_path: Path,
    max_samples: int | None = None,
) -> LabelledDataset:
    """Load the manual evaluation dataset."""
    dataset = LabelledDataset.load_from(manual_dataset_path).filter(
        lambda x: x.label != Label.AMBIGUOUS
    )
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.sample(max_samples)
    return dataset


def run_evaluation_on_manual(
    layer: int,
    model_name: str,
    train_dataset_path: Path,
    manual_dataset_path: Path,
    variation_type: str | None = None,
    variation_value: str | None = None,
    max_samples: int | None = None,
) -> ProbeEvaluationResults:
    """Train a linear probe on our training dataset and evaluate on manual eval dataset."""
    # Load training dataset
    print("Loading training dataset...")
    train_dataset = load_filtered_train_dataset(
        dataset_path=train_dataset_path,
        variation_type=variation_type,
        variation_value=variation_value,
        max_samples=max_samples,
    )

    # Load manual eval dataset
    print("Loading manual evaluation dataset...")
    manual_dataset = load_manual_eval_dataset(
        manual_dataset_path=manual_dataset_path,
        max_samples=max_samples,
    )

    # Create a dictionary with just the manual dataset for evaluation
    eval_datasets = {"manual": manual_dataset}

    # Train probes and evaluate
    results_dict = train_probes_and_save_results(
        model_name=model_name,
        train_dataset=train_dataset,
        train_dataset_path=train_dataset_path,
        eval_datasets=eval_datasets,
        layer=layer,
        output_dir=EVALUATE_PROBES_DIR,
    )

    metrics = []
    dataset_names = []
    for path, (_, results) in results_dict.items():
        print(f"Metrics for {Path(path).stem}: {results.metrics}")
        metrics.append(results)
        dataset_names.append(Path(path).stem)

    # print(f"Metrics: {metrics}")
    # print(f"Results dict: {results_dict}")

    results = ProbeEvaluationResults(
        metrics=metrics,
        datasets=dataset_names,
        train_dataset_path=str(train_dataset_path),
        model_name=model_name,
        variation_type=variation_type,
        variation_value=variation_value,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train linear probes on training dataset and evaluate on manual dataset"
    )
    parser.add_argument(
        "--manual_data",
        type=str,
        default=MANUAL_DATASET_PATH,
        help="Path to manual evaluation data",
    )
    parser.add_argument(
        "--layer", type=int, default=11, help="Layer to extract features from"
    )
    parser.add_argument(
        "--model_name", type=str, default="llama-1b", help="Model name to use"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--variation_type", type=str, default=None, help="Variation type"
    )
    parser.add_argument(
        "--variation_value", type=str, default=None, help="Variation value"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=SYNTHETIC_DATASET_PATH, help="Dataset path"
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    # Create config
    config = EvalRunConfig(
        max_samples=args.max_samples,
        layer=args.layer,
        model_name=LOCAL_MODELS.get(args.model_name, args.model_name),
    )

    # Run evaluation
    results = run_evaluation_on_manual(
        variation_type=args.variation_type,
        variation_value=args.variation_value,
        max_samples=args.max_samples,
        layer=args.layer,
        train_dataset_path=args.dataset_path,
        manual_dataset_path=Path(args.manual_data),
        model_name=config.model_name,
    )

    # Generate output filename
    output_filename = (
        f"manual_eval_{Path(args.manual_data).stem}_{config.output_filename}"
    )

    # Save results
    results.save_to(EVALUATE_PROBES_DIR / output_filename)
    print(f"Results saved to {EVALUATE_PROBES_DIR / output_filename}")
