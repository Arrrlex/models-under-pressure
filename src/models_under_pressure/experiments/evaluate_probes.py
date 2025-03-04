# Code to generate Figure 2
import dataclasses
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import GENERATED_DATASET_TRAIN_TEST_SPLIT, RESULTS_DIR
from models_under_pressure.dataset.loaders import loaders
from models_under_pressure.interfaces.dataset import Dataset
from models_under_pressure.interfaces.results import ProbeEvaluationResults
from models_under_pressure.probes.probes import LinearProbe
from models_under_pressure.scripts.train_probes import (
    load_generated_dataset_split,
    train_probes,
)


def compute_auroc(probe: LinearProbe, dataset: Dataset) -> float:
    """Compute the AUROC score for a probe on a dataset.

    Args:
        probe: A trained probe that implements predict()
        dataset: Dataset to evaluate on

    Returns:
        float: The AUROC score
    """
    # Get activations for the dataset
    activations = probe._llm.get_activations(dataset.inputs, layers=[probe.layer])[0]

    # Get predicted probabilities for the positive class (high stakes)
    y_pred = probe._classifier.predict_proba(
        probe._preprocess_activations(activations)
    )[:, 1]

    # Get true labels
    y_true = dataset.labels_numpy()

    # Compute and return AUROC
    return float(roc_auc_score(y_true, y_pred))


def compute_aurocs(
    train_dataset: Dataset,
    eval_datasets: list[Dataset],
    model_name: str,
    layer: int,
) -> list[float]:
    aurocs = []

    # Train a linear probe on the train dataset
    probes = train_probes(train_dataset, model_name=model_name, layers=[layer])[layer]

    # Evaluate on all eval datasets
    for eval_dataset in eval_datasets:
        aurocs.append(compute_auroc(probes, eval_dataset))

    return aurocs


def run_probe_evaluation(
    layer: int,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    split_path: Path | None = None,
    variation_type: str | None = None,
    variation_value: str | None = None,
    max_samples: int | None = None,
    dataset_path: Path = Path("data/results/prompts_28_02_25.jsonl"),
) -> ProbeEvaluationResults:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    if split_path is None:
        split_path = GENERATED_DATASET_TRAIN_TEST_SPLIT

    # 1. Load train and eval datasets
    train_dataset, _ = load_generated_dataset_split(
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
    eval_dataset_names = ["anthropic", "toolace"]
    eval_datasets = []
    for eval_dataset_name in eval_dataset_names:
        eval_dataset = loaders[eval_dataset_name]()  # type: ignore
        if max_samples is not None:
            indices = np.random.choice(
                range(len(eval_dataset.ids)),
                size=max_samples,
                replace=False,
            )
            eval_dataset = eval_dataset[list(indices)]  # type: ignore
        eval_datasets.append(eval_dataset)

    # Compute AUROCs
    aurocs = compute_aurocs(train_dataset, eval_datasets, model_name, layer)
    for eval_dataset_name, auroc in zip(eval_dataset_names, aurocs):
        print(f"AUROC for {eval_dataset_name}: {auroc}")

    results = ProbeEvaluationResults(
        AUROC=aurocs,
        datasets=eval_dataset_names,
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

    max_samples = 20
    layer = 11
    # variation_type = "prompt_style"
    # variation_type = "language"
    variation_type = None
    variation_value = None  # "Q&A long"
    dataset_path = Path("data/results/prompts_28_02_25.jsonl")
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

    file_name = (
        f"{dataset_path.stem}_{model_name.split('/')[-1]}_{variation_type}_fig2.json"
    )

    results = run_probe_evaluation(
        variation_type=variation_type,
        variation_value=variation_value,
        max_samples=max_samples,
        layer=layer,
        dataset_path=dataset_path,
        model_name=model_name,
    )

    json.dump(dataclasses.asdict(results), open(RESULTS_DIR / file_name, "w"))
