# Code to generate Figure 2
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    EVAL_DATASETS,
    LOCAL_MODELS,
    RESULTS_DIR,
    TRAIN_TEST_SPLIT,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import load_train_test
from models_under_pressure.experiments.train_probes import train_probes_and_save_results
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import ProbeEvaluationResults


def load_eval_datasets(
    max_samples: int | None = None,
) -> dict[str, LabelledDataset]:
    eval_datasets = {}
    for path in EVAL_DATASETS.values():
        dataset = LabelledDataset.load_from(path).filter(
            lambda x: x.label != Label.AMBIGUOUS
        )
        if max_samples:
            dataset = dataset.sample(max_samples)
        eval_datasets[str(path)] = dataset
    return eval_datasets


def run_evaluation(
    layer: int,
    model_name: str,
    split_path: Path | None = None,
    variation_type: str | None = None,
    variation_value: str | None = None,
    max_samples: int | None = None,
    dataset_path: Path = Path("data/results/prompts_28_02_25.jsonl"),
) -> ProbeEvaluationResults:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    if split_path is None:
        split_path = TRAIN_TEST_SPLIT

    # 1. Load train and eval datasets
    train_dataset, _ = load_train_test(
        dataset_path,
        split_path,
    )

    # Filter for one variation type with specific value
    train_dataset = train_dataset.filter(
        lambda x: (
            (
                variation_type is None
                or x.other_fields["variation_type"] == variation_type
            )
            and (
                variation_value is None
                or x.other_fields["variation"] == variation_value
            )
        )
    )

    # Subsample so this runs on the laptop
    if max_samples is not None:
        print("Subsampling the dataset ...")
        indices = np.random.choice(
            range(len(train_dataset.ids)),
            size=max_samples,
            replace=False,
        )
        train_dataset = train_dataset[list(indices)]  # type: ignore

    print(f"Number of samples in train dataset: {len(train_dataset.ids)}")

    # Load eval datasets
    print("Loading eval datasets ...")
    eval_datasets = load_eval_datasets(max_samples=max_samples)

    results_dict = train_probes_and_save_results(
        model_name=model_name,
        train_dataset=train_dataset,
        train_dataset_path=dataset_path,
        eval_datasets=eval_datasets,
        layer=layer,
        output_dir=RESULTS_DIR / "evaluate_probes",
    )

    for path, (_, results) in results_dict.items():
        print(f"Metrics for {Path(path).stem}: {results.metrics}")

    results = ProbeEvaluationResults(
        metrics=[results for _, (_, results) in results_dict.items()],
        train_dataset_path=str(dataset_path),
        datasets=list(eval_datasets.keys()),
        model_name=model_name,
        variation_type=variation_type,
        variation_value=variation_value,
    )
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    config = EvalRunConfig(
        max_samples=None,
        layer=30,
        model_name=LOCAL_MODELS["llama-70b"],
    )

    results = run_evaluation(
        variation_type=config.variation_type,
        variation_value=config.variation_value,
        max_samples=config.max_samples,
        layer=config.layer,
        dataset_path=config.dataset_path,
        split_path=config.split_path,
        model_name=config.model_name,
    )

    output_dir = RESULTS_DIR / "evaluate_probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.save_to(output_dir / config.output_filename)
