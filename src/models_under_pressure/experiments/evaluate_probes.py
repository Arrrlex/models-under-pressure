# Code to generate Figure 2
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    CACHE_DIR,
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    MODEL_MAX_MEMORY,
    TRAIN_DIR,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.experiments.train_probes import (
    evaluate_probe_and_save_results,
)
from models_under_pressure.interfaces.activations import (
    Aggregator,
)
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import EvaluationResult
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import ProbeFactory
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
    config: EvalRunConfig,
    aggregator: Aggregator,
) -> list[EvaluationResult]:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    train_dataset = load_filtered_train_dataset(
        dataset_path=config.dataset_path,
        variation_type=config.variation_type,
        variation_value=config.variation_value,
        max_samples=config.max_samples,
    )

    # Create the model:
    print("Loading model ...")
    model = LLMModel.load(
        config.model_name,
        model_kwargs={
            "device_map": "auto",
            "max_memory": MODEL_MAX_MEMORY[config.model_name],
            "cache_dir": CACHE_DIR,
        },
    )

    # Create the probe:
    print("Creating probe ...")
    probe = ProbeFactory.build(
        probe=config.probe_name,
        model=model,
        train_dataset=train_dataset,
        layer=config.layer,
        output_dir=EVALUATE_PROBES_DIR,
    )

    # Load eval datasets
    print("Loading eval datasets ...")
    eval_datasets = load_eval_datasets(max_samples=config.max_samples)

    results_dict = evaluate_probe_and_save_results(
        model=model,
        probe=probe,
        train_dataset_path=config.dataset_path,
        eval_datasets=eval_datasets,
        layer=config.layer,
        output_dir=EVALUATE_PROBES_DIR,
    )

    # generate calibration plots:
    # run_calibration(
    #     EvalRunConfig(
    #         max_samples=max_samples,
    #         layer=layer,
    #         model_name=model_name,
    #         dataset_path=dataset_path,
    #         variation_type=variation_type,
    #         variation_value=variation_value,
    #     )
    # )

    # Load the ground truth scale labels:
    ground_truth_scale_labels = {}
    ground_truth_labels = {}
    for dataset_name in EVAL_DATASETS.keys():
        data_df = eval_datasets[dataset_name].to_pandas()
        ground_truth_labels[dataset_name] = [
            1 if label == "high-stakes" else 0 for label in data_df["labels"]
        ]
        if dataset_name != "manual":
            ground_truth_scale_labels[dataset_name] = (
                data_df["scale_labels"].astype(int).to_list()
            )
        else:
            ground_truth_scale_labels[dataset_name] = None

    metrics = []
    dataset_names = []
    results_list = []
    column_name_template = f"_{config.model_name.split('/')[-1]}_{config.dataset_path.stem}_l{config.layer}"

    for path, (_, results) in results_dict.items():
        print(f"Metrics for {Path(path).stem}: {results.metrics}")
        metrics.append(results)
        dataset_names.append(Path(path).stem)
        column_name_template = f"_{config.model_name.split('/')[-1]}_{config.dataset_path.stem}_l{config.layer}"

        dataset_results = EvaluationResult(
            config=config,
            metrics=results,
            dataset_name=Path(path).stem,
            method="linear_probe",
            output_scores=results_dict[dataset_names[-1]][0].other_fields[
                f"per_entry_probe_scores{column_name_template}"
            ],  # type: ignore
            output_labels=list(
                int(a > 0.5)
                for a in results_dict[dataset_names[-1]][0].other_fields[
                    f"per_entry_probe_scores{column_name_template}"
                ]
            ),
            ground_truth_scale_labels=ground_truth_scale_labels[dataset_names[-1]],
            ground_truth_labels=ground_truth_labels[dataset_names[-1]],
        )
        results_list.append(dataset_results)
    return results_list


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    config = EvalRunConfig(
        layer=11,
        max_samples=None,
        model_name=LOCAL_MODELS["llama-70b"],
        dataset_path=TRAIN_DIR / "prompts_13_03_25_gpt-4o_filtered.jsonl",
        probe_name="difference_of_means",
    )

    double_check_config(config)

    print(
        f"Running evaluation for {config.id} and results will be saved to {EVALUATE_PROBES_DIR / config.output_filename}"
    )
    results = run_evaluation(
        config=config,
        aggregator=aggregator,
    )

    print(
        f"Saving results for layer {config.layer} to {EVALUATE_PROBES_DIR / config.output_filename}"
    )
    for result in results:
        result.save_to(EVALUATE_PROBES_DIR / config.output_filename)
