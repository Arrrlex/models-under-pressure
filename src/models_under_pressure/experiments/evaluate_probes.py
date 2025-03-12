# Code to generate Figure 2
import dataclasses
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import (
    EVAL_DATASETS,
    LOCAL_MODELS,
    OUTPUT_DIR,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import ProbeEvaluationResults
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe, load_or_train_probe


def compute_auroc(probe: LinearProbe, dataset: LabelledDataset) -> float:
    """Compute the AUROC score for a probe on a dataset.

    Args:
        probe: A trained probe that implements predict()
        dataset: Dataset to evaluate on

    Returns:
        float: The AUROC score
    """
    # Get activations for the dataset
    activations_obj = probe._llm.get_batched_activations(
        dataset=dataset,  # type: ignore
        layer=probe.layer,
    )

    # Get predicted probabilities for the positive class (high stakes)
    processed_activations = probe._preprocess_activations(activations_obj)
    print(f"{processed_activations.shape=}")
    y_pred = probe._classifier.predict_proba(processed_activations)[:, 1]

    # Get true labels
    y_true = dataset.labels_numpy()

    # Compute and return AUROC
    return float(roc_auc_score(y_true, y_pred))


def compute_aurocs(
    train_dataset: LabelledDataset,
    train_dataset_path: Path,
    eval_datasets: dict[str, LabelledDataset],
    model_name: str,
    layer: int,
) -> dict[str, float]:
    model = LLMModel.load(model_name)

    # Train a linear probe on the train dataset
    probe = load_or_train_probe(
        model=model,
        train_dataset=train_dataset,
        train_dataset_path=train_dataset_path,
        layer=layer,
    )

    return {
        name: compute_auroc(probe, eval_dataset)
        for name, eval_dataset in eval_datasets.items()
    }


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
    split_path: Path | None = None,
    variation_type: str | None = None,
    variation_value: str | None = None,
    max_samples: int | None = None,
) -> ProbeEvaluationResults:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    train_dataset = load_filtered_train_dataset(
        dataset_path,
        split_path,
        variation_type,
        variation_value,
        max_samples,
    )

    # Load eval datasets
    print("Loading eval datasets ...")
    eval_datasets = load_eval_datasets(max_samples=max_samples)

    # Compute AUROCs
    aurocs = compute_aurocs(
        train_dataset=train_dataset,
        train_dataset_path=dataset_path,
        eval_datasets=eval_datasets,
        model_name=model_name,
        layer=layer,
    )
    for name, auroc in aurocs.items():
        print(f"AUROC for {name}: {auroc}")

    results = ProbeEvaluationResults(
        AUROC=aurocs,
        train_dataset_path=str(dataset_path),
        datasets=list(eval_datasets.keys()),
        model_name=model_name,
        layer=layer,
        variation_type=variation_type,
        variation_value=variation_value,
    )
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    config = EvalRunConfig(
        max_samples=10,
        layer=11,
        model_name=LOCAL_MODELS["llama-1b"],
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

    json.dump(
        dataclasses.asdict(results),
        open(OUTPUT_DIR / config.output_filename, "w"),
    )
