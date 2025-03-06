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
from models_under_pressure.probes.probes import LinearProbe, compute_accuracy

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
    eval_datasets: dict[str, LabelledDataset],
    layers: list[int],
    output_dir: Path,
) -> None:
    model = LLMModel.load(model_name)
    probes = train_probes(model, train_dataset, layers=layers)

    for layer, probe in tqdm(probes.items(), desc="Processing layers"):
        for eval_dataset_name, eval_dataset in tqdm(
            eval_datasets.items(),
            desc=f"Evaluating datasets for layer {layer}",
            leave=False,
        ):
            probe_scores = probe.per_token_predictions(
                inputs=eval_dataset.inputs,
            )

            probe_scores_list = probe_scores.tolist()

            LabelledDataset(
                inputs=eval_dataset.inputs,
                ids=eval_dataset.ids,
                other_fields={
                    "probe_scores": probe_scores_list,
                    **eval_dataset.other_fields,
                },
            ).save_to(
                output_dir / f"{eval_dataset_name}_withscores_layer_{layer}.jsonl"
            )


if __name__ == "__main__":
    train_dataset = LabelledDataset.load_from(**GENERATED_DATASET)
    eval_datasets = {
        name: LabelledDataset.load_from(**item) for name, item in EVAL_DATASETS.items()
    }
    train_probes_and_save_results(
        model_name=LOCAL_MODELS["llama-8b"],
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
        layers=[7, 10, 12],
        output_dir=Path("data/results/probes"),
    )
