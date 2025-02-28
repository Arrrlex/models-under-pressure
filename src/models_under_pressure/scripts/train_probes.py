import os
from pathlib import Path

import dotenv
import numpy as np

from models_under_pressure.config import (
    ANTHROPIC_SAMPLES_CSV,
    GenerateActivationsConfig,
)
from models_under_pressure.dataset.loaders import load_anthropic_csv
from models_under_pressure.interfaces.dataset import Dataset, Label
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe, compute_accuracy

dotenv.load_dotenv()


def get_activations(
    model: LLMModel,
    config: GenerateActivationsConfig,
    force_recompute: bool = False,
) -> np.ndarray:
    assert model.name == config.model_name

    if config.output_file.exists() and not force_recompute:
        return np.load(config.output_file)["activations"]
    else:
        print("Generating activations...")
        activations = model.get_activations(
            inputs=config.dataset.inputs, layers=[config.layer]
        )[0]
        np.savez_compressed(config.output_file, activations=activations)
        return activations


def test_get_activations(config: GenerateActivationsConfig, model_name: str):
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )
    activations = get_activations(model=model, config=config, force_recompute=True)
    # Load precomputed activations
    activations2 = get_activations(
        model=model,
        config=config,
        force_recompute=False,
    )
    assert np.allclose(activations, activations2)


def test_compute_accuracy(
    model_name: str,
    config: GenerateActivationsConfig,
):
    print("Loading model...")
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )

    print("Loading activations...")
    activations = get_activations(
        model=model,
        config=config,
    )

    if any(label == Label.AMBIGUOUS for label in dataset.labels):
        raise ValueError("Dataset contains ambiguous labels")

    probe = LinearProbe(_llm=model, layer=config.layer)

    print("Training probe...")
    probe.fit(X=activations, y=dataset.labels_numpy())

    print("Computing accuracy...")
    accuracy = compute_accuracy(probe, dataset, activations)
    print(f"Accuracy: {accuracy}")


def train_probes(dataset_path: Path, model_name: str) -> list[LinearProbe]:
    pass


def create_train_test_split(dataset: Dataset) -> tuple[Dataset, Dataset]:
    pass


def create_cross_validation_splits(dataset: Dataset) -> list[Dataset]:
    pass


def cross_validate_probe(
    probe: LinearProbe, dataset_splits: list[Dataset]
) -> np.ndarray:
    activations = get_activations(
        model=probe._llm,
        config=config,
    )
    accuracies = np.array(
        [compute_accuracy(probe, dataset, activations) for dataset in dataset_splits]
    )
    return np.mean(accuracies, axis=1)


def cross_validate_probes(probes: list[LinearProbe], dataset: Dataset) -> np.ndarray:
    dataset_splits = create_cross_validation_splits(dataset)
    accuracies = np.array(
        [cross_validate_probe(probe, dataset_splits) for probe in probes]
    )
    return np.mean(accuracies, axis=1)


if __name__ == "__main__":
    dataset_path = ANTHROPIC_SAMPLES_CSV
    layer = 10
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print("Loading dataset...")
    dataset = load_anthropic_csv(dataset_path)

    config = GenerateActivationsConfig(
        dataset=dataset, model_name=model_name, layer=layer
    )

    test_get_activations(
        config=config,
        model_name=model_name,
    )
    test_compute_accuracy(
        model_name=model_name,
        config=config,
    )
