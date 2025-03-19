# Code to generate Figure 2
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    EVAL_DATASETS,
    LOCAL_MODELS,
    OUTPUT_DIR,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.experiments.train_probes import train_probes_and_save_results
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import EvaluationResult
from models_under_pressure.utils import double_check_config


def load_eval_datasets(
    max_samples: int | None = None,
) -> dict[str, LabelledDataset]:
    eval_datasets = {}
    for name, path in EVAL_DATASETS.items():
        dataset = LabelledDataset.load_from(path).filter(
            lambda x: x.label != Label.AMBIGUOUS
        )
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.sample(max_samples)
        eval_datasets[name] = dataset
    return eval_datasets


def run_evaluation(
    layer: int,
    model_name: str,
    dataset_path: Path,
    variation_type: str | None = None,
    variation_value: str | None = None,
    max_samples: int | None = None,
) -> list[EvaluationResult]:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    train_dataset = load_filtered_train_dataset(
        dataset_path=dataset_path,
        variation_type=variation_type,
        variation_value=variation_value,
        max_samples=max_samples,
    )

    # Load eval datasets
    print("Loading eval datasets ...")
    eval_datasets = load_eval_datasets(max_samples=max_samples)

    results_dict = train_probes_and_save_results(
        model_name=model_name,
        train_dataset=train_dataset,
        train_dataset_path=dataset_path,
        eval_datasets=eval_datasets,
        layer=layer,
        output_dir=OUTPUT_DIR,
    )
    metrics = []
    dataset_names = []
    results_list = []
    column_name_template = f"_{model_name.split('/')[-1]}_{dataset_path.stem}_l{layer}"

    for path, (_, results) in results_dict.items():
        print(f"Metrics for {Path(path).stem}: {results.metrics}")
        metrics.append(results)
        dataset_names.append(Path(path).stem)
        column_name_template = (
            f"_{model_name.split('/')[-1]}_{dataset_path.stem}_l{layer}"
        )

        dataset_results = EvaluationResult(
            metrics=results,
            dataset_name=Path(path).stem,
            model_name=model_name,
            train_dataset_path=str(dataset_path),
            variation_type=variation_type,
            variation_value=variation_value,
            method="linear_probe",
            method_details={"layer": layer},
            train_dataset_details={"max_samples": max_samples},
            eval_dataset_details={"max_samples": max_samples},
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


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    configs = [
        EvalRunConfig(
            layer=layer,
            max_samples=None,
            model_name=LOCAL_MODELS["llama-70b"],
        )
        for layer in [11, 22, 33, 44, 55, 66, 77]
    ]

    double_check_config(configs)

    for config in configs:
        results = run_evaluation(
            variation_type=config.variation_type,
            variation_value=config.variation_value,
            max_samples=config.max_samples,
            layer=config.layer,
            dataset_path=config.dataset_path,
            model_name=config.model_name,
        )

        print(
            f"Saving results for layer {config.layer} to {OUTPUT_DIR / config.output_filename}"
        )
        for result in results:
            result.save_to(OUTPUT_DIR / config.output_filename)
