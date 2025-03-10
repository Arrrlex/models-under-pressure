from pathlib import Path

import dotenv
import numpy as np
from tqdm import tqdm

from models_under_pressure.config import (
    EVAL_DATASETS,
    GENERATED_DATASET,
    LOCAL_MODELS,
)
from models_under_pressure.experiments.dataset_splitting import (
    create_cross_validation_splits,
)
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import (
    LinearProbe,
    compute_accuracy,
    load_or_train_probe,
)

# Set random seed for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

dotenv.load_dotenv()


def train_probes(
    model: LLMModel, dataset: LabelledDataset, layers: list[int] | None = None
) -> dict[int, LinearProbe]:
    """Train a probe for each layer in the model."""

    layers = layers or list(range(model.n_layers))

    if any(label == Label.AMBIGUOUS for label in dataset.labels):
        raise ValueError("Training dataset contains ambiguous labels")

    # Iterate over layers. For each layer, create a config, then train a probe and store it
    return {
        layer: LinearProbe(_llm=model, layer=layer).fit(dataset)
        for layer in tqdm(layers, desc="Training probes")
    }


def cross_validate_probe(
    probe: LinearProbe, dataset_splits: list[LabelledDataset]
) -> np.ndarray:
    accuracies = []

    for i, dataset in enumerate(dataset_splits):
        activations_obj = probe._llm.get_batched_activations(
            dataset=dataset,
            layer=probe.layer,
        )
        accuracy = compute_accuracy(
            probe,
            dataset,
            activations_obj,
        )
        accuracies.append(accuracy)

    return np.mean(np.array(accuracies), axis=1)


def cross_validate_probes(
    probes: list[LinearProbe], dataset: LabelledDataset
) -> np.ndarray:
    dataset_splits = create_cross_validation_splits(dataset)
    accuracies = np.array(
        [cross_validate_probe(probe, dataset_splits) for probe in probes]
    )
    return np.mean(accuracies, axis=1)


def train_probes_and_save_results(
    model_name: str,
    train_dataset: LabelledDataset,
    train_dataset_path: Path,
    eval_datasets: dict[str, LabelledDataset],
    layers: list[int],
    output_dir: Path,
) -> None:
    model = LLMModel.load(model_name)
    probes = {
        layer: load_or_train_probe(model, train_dataset, train_dataset_path, layer)
        for layer in layers
    }

    probe_scores_dict = {name: {} for name in eval_datasets.keys()}

    for layer, probe in tqdm(probes.items(), desc="Processing layers"):
        for eval_dataset_name, eval_dataset in tqdm(
            eval_datasets.items(),
            desc=f"Evaluating datasets for layer {layer}",
            leave=False,
        ):
            probe_scores = probe.per_token_predictions(
                inputs=eval_dataset.inputs,
            )

            probe_logits = np.log(probe_scores / (1 - probe_scores))

            probe_logits_list = probe_logits.tolist()

            probe_scores_dict[eval_dataset_name][layer] = probe_logits_list

    for eval_dataset_name in eval_datasets.keys():
        dataset_with_probe_scores = LabelledDataset.load_from(
            output_dir / f"{eval_dataset_name}.jsonl"
        )
        extra_fields = {}
        for layer, probe_logits in probe_scores_dict[eval_dataset_name].items():
            col_name = f"probe_logits_{model.name.split('/')[-1]}_{train_dataset_path.stem}_l{layer}"
            extra_fields[col_name] = probe_logits
        dataset_with_probe_scores.other_fields = {
            **dataset_with_probe_scores.other_fields,
            **extra_fields,
        }
        dataset_with_probe_scores.save_to(
            output_dir / f"{eval_dataset_name}.jsonl", overwrite=True
        )


if __name__ == "__main__":
    train_dataset = LabelledDataset.load_from(**GENERATED_DATASET)
    eval_datasets = {
        name: LabelledDataset.load_from(path) for name, path in EVAL_DATASETS.items()
    }
    train_probes_and_save_results(
        model_name=LOCAL_MODELS["llama-8b"],
        train_dataset=train_dataset,
        train_dataset_path=GENERATED_DATASET["file_path_or_name"],
        eval_datasets=eval_datasets,
        layers=[7, 10, 12],
        output_dir=Path("data/results/train_probes"),
    )
