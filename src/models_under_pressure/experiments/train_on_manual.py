# Script to train on manual dataset and evaluate on eval datasets
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    GENERATED_DATASET_PATH,
    LOCAL_MODELS,
    MANUAL_DATASET_PATH,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_train_test,
)
from models_under_pressure.experiments.train_probes import train_probes_and_save_results
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import ProbeEvaluationResults


def load_manual_dataset(
    max_samples: int | None = None,
) -> LabelledDataset:
    """Load the manual dataset for training."""
    dataset = LabelledDataset.load_from(MANUAL_DATASET_PATH).filter(
        lambda x: x.label != Label.AMBIGUOUS
    )
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.sample(max_samples)
    return dataset


def load_eval_datasets(
    max_samples: int | None = None,
) -> dict[str, LabelledDataset]:
    """Load evaluation datasets."""
    eval_datasets = {}
    for name, path in EVAL_DATASETS.items():
        dataset = LabelledDataset.load_from(path).filter(
            lambda x: x.label != Label.AMBIGUOUS
        )
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.sample(max_samples)
        eval_datasets[name] = dataset
    return eval_datasets


def run_manual_evaluation(
    layer: int,
    model_name: str,
    max_samples: int | None = None,
) -> ProbeEvaluationResults:
    """Train a linear probe on the manual dataset and evaluate on all eval datasets."""
    # Load manual dataset for training
    print("Loading manual dataset for training...")
    train_dataset = load_manual_dataset(max_samples=max_samples)

    # Load eval datasets
    print("Loading eval datasets...")
    eval_datasets = load_eval_datasets(max_samples=max_samples)

    results_dict = train_probes_and_save_results(
        model_name=model_name,
        train_dataset=train_dataset,
        train_dataset_path=MANUAL_DATASET_PATH,
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

    results = ProbeEvaluationResults(
        metrics=metrics,
        datasets=dataset_names,
        train_dataset_path=str(MANUAL_DATASET_PATH),
        model_name=model_name,
        variation_type=None,
        variation_value=None,
    )
    return results


def evaluate_on_train_test_split(
    layer: int,
    model_name: str,
    dataset_path: Path,
    is_test: bool = False,
    max_samples: int | None = None,
) -> ProbeEvaluationResults:
    """Train a linear probe on the manual dataset and evaluate on a train/test split.

    Args:
        layer: Layer to extract embeddings from
        model_name: Name of the model to use
        dataset_path: Path to the dataset to evaluate on
        is_test: If True, evaluate on test split, otherwise on train split
        max_samples: Maximum number of samples to use

    Returns:
        Evaluation results
    """
    # Load manual dataset for training
    print("Loading manual dataset for training...")
    train_dataset = load_manual_dataset(max_samples=max_samples)

    # Load the train/test split using the existing function
    train_split, test_split = load_train_test(dataset_path)

    # Select which split to use for evaluation
    eval_dataset = test_split if is_test else train_split

    # Filter out ambiguous examples
    eval_dataset = eval_dataset.filter(lambda x: x.label != Label.AMBIGUOUS)

    # Subsample if needed
    if max_samples and len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.sample(max_samples)

    split_name = "test" if is_test else "train"
    dataset_name = Path(dataset_path).stem
    eval_datasets = {f"{dataset_name}_{split_name}": eval_dataset}

    results_dict = train_probes_and_save_results(
        model_name=model_name,
        train_dataset=train_dataset,
        train_dataset_path=MANUAL_DATASET_PATH,
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

    results = ProbeEvaluationResults(
        metrics=metrics,
        datasets=dataset_names,
        train_dataset_path=str(MANUAL_DATASET_PATH),
        model_name=model_name,
        variation_type=None,
        variation_value=None,
    )
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    config = EvalRunConfig(
        max_samples=100,
        layer=11,
        model_name=LOCAL_MODELS["llama-1b"],
    )

    # Choose which evaluation to run
    evaluation_type = "standard"  # Options: "standard", "train", "test"

    if evaluation_type == "standard":
        # Evaluate on all eval datasets
        results = run_manual_evaluation(
            max_samples=config.max_samples,
            layer=config.layer,
            model_name=config.model_name,
        )
        output_filename = (
            f"manual_train_{Path(config.model_name).stem}_layer{config.layer}.json"
        )

    elif evaluation_type in ["train", "test"]:
        # Evaluate on train or test split of a specific dataset
        is_test = evaluation_type == "test"
        dataset_path = GENERATED_DATASET_PATH
        results = evaluate_on_train_test_split(
            layer=config.layer,
            model_name=config.model_name,
            dataset_path=dataset_path,
            is_test=is_test,
            max_samples=config.max_samples,
        )
        split_name = "test" if is_test else "train"
        dataset_name = dataset_path.stem
        output_filename = f"manual_train_{Path(config.model_name).stem}_eval_{dataset_name}_{split_name}_layer{config.layer}_fig2.json"

    # Save results
    results.save_to(EVALUATE_PROBES_DIR / output_filename)
    print(f"Results saved to {EVALUATE_PROBES_DIR / output_filename}")
