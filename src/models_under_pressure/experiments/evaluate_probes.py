# Code to generate Figure 2
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    CACHE_DIR,
    EVAL_DATASETS,
    LOCAL_MODELS,
    MODEL_MAX_MEMORY,
    OUTPUT_DIR,
    EvalRunConfig,
    TRAIN_DIR,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.experiments.train_probes import (
    evaluate_probe_and_save_results,
)
from models_under_pressure.interfaces.activations import (
    Aggregator,
    Postprocessors,
    Preprocessors,
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
    layer: int,
    model_name: str,
    probe_name: str,
    dataset_path: Path,
    aggregator: Aggregator,
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

    # Create the model:
    print("Loading model ...")
    model = LLMModel.load(
        model_name,
        model_kwargs={
            "device_map": "auto",
            "max_memory": MODEL_MAX_MEMORY[model_name],
            "cache_dir": CACHE_DIR,
        },
    )

    # Create the probe:
    print("Creating probe ...")
    probe = ProbeFactory.build(
        probe=probe_name,
        model=model,
        train_dataset=train_dataset,
        layer=layer,
        aggregator=aggregator,
        output_dir=OUTPUT_DIR,
    )

    # Load eval datasets
    print("Loading eval datasets ...")
    eval_datasets = load_eval_datasets(max_samples=max_samples)

    results_dict = evaluate_probe_and_save_results(
        model=model,
        probe=probe,
        train_dataset_path=dataset_path,
        eval_datasets=eval_datasets,
        layer=layer,
        output_dir=OUTPUT_DIR,
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
            max_samples=10,
            variation_type="language",
            variation_value="English",
            model_name=LOCAL_MODELS["llama-70b"],
            dataset_path=TRAIN_DIR / "prompts_13_03_25_gpt-4o_filtered.jsonl",
        )
        for layer in [31]
    ]

    aggregator = Aggregator(
        preprocessor=Preprocessors.mean,
        postprocessor=Postprocessors.sigmoid,
    )

    double_check_config(configs)

    for config in configs:
        results = run_evaluation(
            probe_name=config.probe_name,
            variation_type=config.variation_type,
            variation_value=config.variation_value,
            max_samples=config.max_samples,
            layer=config.layer,
            dataset_path=config.dataset_path,
            model_name=config.model_name,
            aggregator=aggregator,
        )

        print(
            f"Saving results for layer {config.layer} to {OUTPUT_DIR / config.output_filename}"
        )
        for result in results:
            result.save_to(OUTPUT_DIR / config.output_filename)
