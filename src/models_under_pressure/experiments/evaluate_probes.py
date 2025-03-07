# Code to generate Figure 2
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import (
    EVAL_DATASETS,
    OUTPUT_DIR,
    TRAIN_TEST_SPLIT,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import load_train_test
from models_under_pressure.experiments.train_probes import train_probes
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import ProbeEvaluationResults
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe


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
        dataset=dataset,
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
    eval_datasets: list[LabelledDataset],
    model_name: str,
    layer: int,
) -> list[float]:
    aurocs = []
    model = LLMModel.load(model_name, model_kwargs={"torch_dtype": torch.float16})
    # Train a linear probe on the train dataset
    probes = train_probes(model, train_dataset, layers=[layer])[layer]

    torch.cuda.empty_cache()

    # Evaluate on all eval datasets
    for eval_dataset in eval_datasets:
        aurocs.append(compute_auroc(probes, eval_dataset))
        torch.cuda.empty_cache()

    return aurocs


def load_eval_datasets(
    max_samples: int | None = None,
) -> tuple[list[LabelledDataset], list[str]]:
    eval_datasets = []
    eval_dataset_names = []
    for eval_dataset_name, eval_dataset_config in EVAL_DATASETS.items():
        eval_dataset = LabelledDataset.load_from(**eval_dataset_config).filter(
            lambda x: x.label != Label.AMBIGUOUS
        )
        if max_samples is not None:
            indices = np.random.choice(
                range(len(eval_dataset.ids)),
                size=max_samples,
                replace=False,
            )
            eval_dataset = eval_dataset[list(indices)]  # type: ignore
        eval_datasets.append(eval_dataset)
        eval_dataset_names.append(eval_dataset_name)
    return eval_datasets, eval_dataset_names


def run_evaluation(
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
    eval_datasets, eval_dataset_names = load_eval_datasets(max_samples=max_samples)

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

    config = EvalRunConfig(
        max_samples=None,
        layer=11,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
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
        dataclasses.asdict(results), open(OUTPUT_DIR / config.output_filename, "w")
    )
