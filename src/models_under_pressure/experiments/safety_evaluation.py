import dataclasses
import json
from pathlib import Path

import numpy as np

from models_under_pressure.config import (
    AIS_DATASETS,
    DEFAULT_GPU_MODEL,
    DEFAULT_OTHER_MODEL,
    DEVICE,
    OUTPUT_DIR,
    SafetyRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.experiments.evaluate_probes import (
    ProbeEvaluationResults,
    compute_aurocs,
)
from models_under_pressure.interfaces.dataset import LabelledDataset


def run_safety_evaluation(
    layer: int,
    model_name: str = DEFAULT_GPU_MODEL if "cuda" in DEVICE else DEFAULT_OTHER_MODEL,
    split_path: Path | None = None,
    variation_type: str | None = None,
    variation_value: str | None = None,
    max_samples: int | None = None,
    dataset_path: Path = Path("data/results/prompts_28_02_25.jsonl"),
) -> ProbeEvaluationResults:
    """Compute AUROCs for figure 1."""
    train_dataset = load_filtered_train_dataset(
        dataset_path,
        split_path,
        variation_type,
        variation_value,
        max_samples,
    )

    # Load eval datasets
    print("Loading eval datasets ...")
    eval_datasets = []
    eval_dataset_names = []

    # Sandbagging dataset
    # TODO Make it possible to evaluate on high-stakes label as well
    sandbagging_dataset = LabelledDataset.load_from(
        **AIS_DATASETS["mmlu_sandbagging"],
    )
    deception_dataset = LabelledDataset.load_from(
        **AIS_DATASETS["deception"],
    )
    if max_samples is not None:
        print("Subsampling the dataset ...")
        indices_sandbagging = np.random.choice(
            range(len(sandbagging_dataset.ids)),
            size=max_samples,
            replace=False,
        )
        indices_deception = np.random.choice(
            range(len(deception_dataset.ids)),
            size=max_samples,
            replace=False,
        )
        sandbagging_dataset = sandbagging_dataset[list(indices_sandbagging)]  # type: ignore
        deception_dataset = deception_dataset[list(indices_deception)]  # type: ignore
    eval_datasets.append(sandbagging_dataset)
    eval_dataset_names.append("Sandbagging")
    eval_datasets.append(deception_dataset)
    eval_dataset_names.append("Deception")

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

    config = SafetyRunConfig(
        max_samples=100,
        layer=11,
    )
    results = run_safety_evaluation(
        variation_type="prompt_style",
        variation_value="Third Person",
        max_samples=config.max_samples,
        layer=config.layer,
        dataset_path=config.dataset_path,
        split_path=config.split_path,
        model_name=config.model_name,
    )

    json.dump(
        dataclasses.asdict(results), open(OUTPUT_DIR / config.output_filename, "w")
    )
