# Script to train on manual dataset and evaluate on eval datasets
import argparse
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    MANUAL_DATASET_PATH,
    MANUAL_UPSAMPLED_DATASET_PATH,
    SYNTHETIC_DATASET_PATH,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_train_test,
)
from models_under_pressure.experiments.train_probes import train_probes_and_save_results
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import EvaluationResult
from models_under_pressure.utils import double_check_config


def load_manual_dataset(
    max_samples: int | None = None,
    upsampled: bool = False,
) -> LabelledDataset:
    """Load the manual dataset for training."""
    if upsampled:
        dataset_path = MANUAL_UPSAMPLED_DATASET_PATH
    else:
        dataset_path = MANUAL_DATASET_PATH
    dataset = LabelledDataset.load_from(dataset_path).filter(
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
    train_dataset: LabelledDataset,
    train_dataset_path: Path,
    max_samples: int | None = None,
) -> list[EvaluationResult]:
    """Train a linear probe on the specified dataset and evaluate on all eval datasets."""
    # Load eval datasets
    print("Loading eval datasets...")
    eval_datasets = load_eval_datasets(max_samples=max_samples)

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
    results_list = []
    for path, (_, results) in results_dict.items():
        print(f"Metrics for {Path(path).stem}: {results.metrics}")
        metrics.append(results)
        dataset_names.append(Path(path).stem)
        column_name_template = f"_{model_name.split('/')[-1]}_manual_l{layer}"

        dataset_results = EvaluationResult(
            metrics=results,
            dataset_name=Path(path).stem,
            model_name=model_name,
            train_dataset_path=str(train_dataset_path),
            variation_type=None,
            variation_value=None,
            method="linear_probe_manual",
            method_details={"layer": layer},
            train_dataset_details={"max_samples": max_samples},
            eval_dataset_details={"max_samples": max_samples},
            output_scores=results_dict[dataset_names[-1]][0].other_fields[
                f"per_entry_probe_scores{column_name_template}"
            ],  # type: ignore
        )
        results_list.append(dataset_results)
    return results_list


def evaluate_on_train_test_split(
    layer: int,
    model_name: str,
    train_dataset_path: Path,
    eval_dataset_path: Path,
    is_test: bool = False,
    max_samples: int | None = None,
) -> list[EvaluationResult]:
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
    train_split, test_split = load_train_test(eval_dataset_path)

    # Select which split to use for evaluation
    eval_dataset = test_split if is_test else train_split

    # Filter out ambiguous examples
    eval_dataset = eval_dataset.filter(lambda x: x.label != Label.AMBIGUOUS)

    # Subsample if needed
    if max_samples and len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.sample(max_samples)

    split_name = "test" if is_test else "train"
    eval_dataset_name = Path(eval_dataset_path).stem
    eval_datasets = {f"{eval_dataset_name}_{split_name}": eval_dataset}

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
    results_list = []
    for path, (_, results) in results_dict.items():
        print(f"Metrics for {Path(path).stem}: {results.metrics}")
        metrics.append(results)
        dataset_names.append(Path(path).stem)

        column_name_template = f"_{model_name.split('/')[-1]}_manual_l{layer}"
        dataset_name = (
            eval_dataset_name  # + "_test" if is_test else eval_dataset_name + "_train"
        )
        breakpoint()
        dataset_results = EvaluationResult(
            dataset_name=dataset_name,
            model_name=model_name,
            train_dataset_path=str(train_dataset_path),
            metrics=results,
            method="linear_probe_manual",
            method_details={"layer": layer},
            train_dataset_details={"max_samples": max_samples},
            eval_dataset_details={"max_samples": max_samples, "split": split_name},
            output_scores=results_dict[dataset_names[-1]][0].other_fields[
                f"per_entry_probe_scores{column_name_template}"
            ],  # type: ignore
            output_labels=list(
                int(a > 0.5)
                for a in results_dict[dataset_names[-1]][0].other_fields[
                    f"per_entry_probe_scores{column_name_template}"
                ]
            ),  # type: ignore
        )
        results_list.append(dataset_results)
    return results_list


def main(
    config: EvalRunConfig,
    evaluation_type: str,
    dataset_path: Path | None = None,
    train_dataset_type: str = "manual",
):
    # Load training dataset based on type
    if train_dataset_type == "manual":
        train_dataset = load_manual_dataset(max_samples=config.max_samples)
        train_dataset_path = MANUAL_DATASET_PATH
    elif train_dataset_type == "upsampled":
        train_dataset = load_manual_dataset(
            max_samples=config.max_samples, upsampled=True
        )
        train_dataset_path = MANUAL_UPSAMPLED_DATASET_PATH
    else:
        raise ValueError(f"Invalid train dataset type: {train_dataset_type}")

    if evaluation_type == "standard":
        # Evaluate on all eval datasets
        results = run_manual_evaluation(
            max_samples=config.max_samples,
            layer=config.layer,
            model_name=config.model_name,
            train_dataset=train_dataset,
            train_dataset_path=train_dataset_path,
        )
        output_filename = f"{train_dataset_type}_train_{Path(config.model_name).stem}_layer{config.layer}_fig2.json"

    elif evaluation_type in ["train", "test"]:
        # Evaluate on train or test split of a specific dataset
        is_test = evaluation_type == "test"
        dataset_path = (
            Path(args.dataset_path) if args.dataset_path else SYNTHETIC_DATASET_PATH
        )
        results = evaluate_on_train_test_split(
            layer=config.layer,
            model_name=config.model_name,
            train_dataset_path=train_dataset_path,
            eval_dataset_path=dataset_path,
            is_test=is_test,
            max_samples=config.max_samples,
        )
        split_name = "test" if is_test else "train"
        dataset_name = dataset_path.stem
        output_filename = f"manual_train_{Path(config.model_name).stem}_eval_{dataset_name}_{split_name}_layer{config.layer}_fig2.json"

        # Save resultswe    for result in results:
        # result.save_to(EVALUATE_PROBES_DIR / output_filename)
        for result in results:
            result.save_to(EVALUATE_PROBES_DIR / output_filename)
    print(f"Results saved to {EVALUATE_PROBES_DIR / output_filename}")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Train on manual dataset and evaluate on eval datasets"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--layer", type=int, default=11, help="Layer to extract embeddings from"
    )
    parser.add_argument(
        "--model_name", type=str, default="llama-1b", help="Model name to use"
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="standard",
        choices=["standard", "train", "test"],
        help="Type of evaluation to run",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset for train/test evaluation (defaults to GENERATED_DATASET_PATH)",
    )
    parser.add_argument(
        "--train_dataset_type",
        type=str,
        default="manual",
        choices=["manual", "upsampled"],
        help="Type of training dataset to use",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    config = EvalRunConfig(
        max_samples=args.max_samples,
        layer=args.layer,
        model_name=LOCAL_MODELS.get(args.model_name, args.model_name),
    )
    double_check_config(config)
    main(config, args.evaluation_type, args.dataset_path, args.train_dataset_type)
