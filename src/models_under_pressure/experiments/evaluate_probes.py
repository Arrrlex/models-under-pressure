# Code to generate Figure 2
import json
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.experiments.train_probes import (
    evaluate_probe_and_save_results,
)
from models_under_pressure.interfaces.dataset import DatasetSpec, Label, LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.interfaces.results import EvaluationResult
from models_under_pressure.probes.probes import ProbeFactory
from models_under_pressure.utils import double_check_config


def load_eval_datasets(
    use_test_set: bool,
    max_samples: int | None = None,
) -> tuple[dict[str, LabelledDataset], dict[str, Path]]:
    eval_datasets = {}
    eval_dataset_paths = {}
    # max_samples = 200
    # datasets = TEST_DATASETS if use_test_set else EVAL_DATASETS
    datasets = {"anthropic": EVAL_DATASETS["anthropic"]}
    for name, path in datasets.items():
        dataset = LabelledDataset.load_from(path).filter(
            lambda x: x.label != Label.AMBIGUOUS
        )
        if max_samples and len(dataset) > max_samples:
            # Sample equal number of high and low stakes examples
            high_stakes = dataset.filter(lambda x: x.label == Label.HIGH_STAKES)
            low_stakes = dataset.filter(lambda x: x.label == Label.LOW_STAKES)
            n_per_class = min(len(high_stakes), len(low_stakes), max_samples // 2)
            dataset = LabelledDataset.concatenate(
                [high_stakes.sample(n_per_class), low_stakes.sample(n_per_class)]
            )
        eval_datasets[name] = dataset
        eval_dataset_paths[name] = path
    return eval_datasets, eval_dataset_paths


def run_evaluation(
    config: EvalRunConfig,
) -> tuple[list[EvaluationResult], list[float]]:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    train_dataset = load_filtered_train_dataset(
        dataset_spec=config.dataset_spec,
        variation_type=config.variation_type,
        variation_value=config.variation_value,
        max_samples=config.max_samples,
    )

    # Create the probe:
    print("Creating probe ...")
    probe = ProbeFactory.build(
        probe_spec=config.probe_spec,
        model_name=config.model_name,
        layer=config.layer,
        train_dataset=train_dataset,
    )

    # Load eval datasets
    print("Loading eval datasets ...")
    eval_datasets, eval_dataset_paths = load_eval_datasets(
        use_test_set=config.use_test_set,
        max_samples=config.max_samples,
    )

    train_dataset = LabelledDataset.load_from(config.dataset_spec)

    results_dict, coefs = evaluate_probe_and_save_results(
        probe=probe,
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
        layer=config.layer,
        output_dir=EVALUATE_PROBES_DIR,
        model_name=config.model_name,
    )

    # Load the ground truth scale labels:
    ground_truth_scale_labels = {}
    ground_truth_labels = {}
    for dataset_name in eval_datasets.keys():
        data_df = eval_datasets[dataset_name].to_pandas()
        ground_truth_labels[dataset_name] = eval_datasets[dataset_name].labels_numpy()

        if dataset_name == "manual":
            ground_truth_scale_labels[dataset_name] = None
        else:
            ground_truth_scale_labels[dataset_name] = (
                data_df["scale_labels"].astype(int).to_list()
            )

    metrics = []
    dataset_names = []
    results_list = []

    for path, (probe_scores, results) in results_dict.items():
        print(f"Metrics for {Path(path).stem}: {results.metrics}")
        metrics.append(results)
        dataset_names.append(Path(path).stem)

        dataset_results = EvaluationResult(
            config=config,
            metrics=results,
            dataset_name=Path(path).stem,
            method="linear_probe",
            output_scores=probe_scores,
            output_labels=list(int(a > 0.5) for a in probe_scores),
            ground_truth_scale_labels=ground_truth_scale_labels[dataset_names[-1]],
            ground_truth_labels=ground_truth_labels[dataset_names[-1]],
            dataset_path=eval_dataset_paths[dataset_names[-1]],
        )

        results_list.append(dataset_results)

    return results_list, coefs


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    config = EvalRunConfig(
        layer=5,
        max_samples=None,
        model_name=LOCAL_MODELS["llama-1b"],
        dataset_spec=DatasetSpec(path=SYNTHETIC_DATASET_PATH),
        probe_spec=ProbeSpec(
            name="pytorch_per_token_probe",
            hyperparams={"batch_size": 16, "epochs": 3, "device": "cpu"},
        ),
    )

    double_check_config(config)

    print(f"Running probe evaluation with ID {config.id}")
    print(f"Results will be saved to {EVALUATE_PROBES_DIR / config.output_filename}")
    results, coefs = run_evaluation(config=config)

    print(f"Saving results to {EVALUATE_PROBES_DIR / config.output_filename}")
    for result in results:
        result.save_to(EVALUATE_PROBES_DIR / config.output_filename)

    coefs_dict = {
        "id": config.id,
        "coefs": coefs[0].tolist(),  # type: ignore
    }
    with open(EVALUATE_PROBES_DIR / config.coefs_filename, "w") as f:
        json.dump(coefs_dict, f)
