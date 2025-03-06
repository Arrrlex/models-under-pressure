import dotenv
import numpy as np

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
    probes = {}
    for layer in layers:
        print("Loading training activations...")
        probe = LinearProbe(_llm=model, layer=layer)
        print("Training probe...")
        probe.fit(dataset)
        probes[layer] = probe
    return probes


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
