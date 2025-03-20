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
from models_under_pressure.interfaces.results import ProbeEvaluationResults
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import ProbeFactory


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
) -> ProbeEvaluationResults:
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
    metrics = []
    dataset_names = []
    for path, (_, results) in results_dict.items():
        print(f"Metrics for {Path(path).stem}: {results.metrics}")
        metrics.append(results)
        dataset_names.append(Path(path).stem)

    results = ProbeEvaluationResults(
        metrics=metrics,
        datasets=dataset_names,
        train_dataset_path=str(dataset_path),
        model_name=model_name,
        variation_type=variation_type,
        variation_value=variation_value,
    )
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    for layer in [11, 22, 33, 44, 55, 66, 77]:
        config = EvalRunConfig(
            max_samples=None,
            layer=layer,
            model_name=LOCAL_MODELS["llama-70b"],
        )

        aggregator = Aggregator(
            preprocessor=Preprocessors.mean,
            postprocessor=Postprocessors.sigmoid,
        )

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
            f"Saving results for layer {layer} to {OUTPUT_DIR / config.output_filename}"
        )
        results.save_to(OUTPUT_DIR / config.output_filename)
