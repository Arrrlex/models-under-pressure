import os

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
    train_config: GenerateActivationsConfig,
    test_config: GenerateActivationsConfig,
):
    print("Loading model...")
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )

    print("Loading training activations...")
    activations = get_activations(
        model=model,
        config=train_config,
    )
    if any(label == Label.AMBIGUOUS for label in train_config.dataset.labels):
        raise ValueError("Training dataset contains ambiguous labels")

    probe = LinearProbe(_llm=model, layer=train_config.layer)

    print("Training probe...")
    probe.fit(X=activations, y=train_config.dataset.labels_numpy())

    print("Loading testing activations...")
    activations = get_activations(
        model=model,
        config=test_config,
    )
    if any(label == Label.AMBIGUOUS for label in test_config.dataset.labels):
        raise ValueError("Test dataset contains ambiguous labels")

    print("Computing accuracy...")
    accuracy = compute_accuracy(probe, test_config.dataset, activations)
    print(f"Accuracy: {accuracy}")


# Prios:
# 1. Generate heatmap for prompt style using train-test split
# 2. Generate heatmap for tone using train-test split
# 1. Generate heatmap for language using train-test split


def train_probes(dataset: Dataset, model_name: str) -> list[LinearProbe]:
    pass


def create_train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    split_field: str | None = None,
) -> tuple[Dataset, Dataset]:
    """Create a train-test split of the dataset.

    Args:
        dataset: Dataset to split
        test_size: Fraction of data to use for test set
        split_field: If provided, ensures examples with the same value for this field
                    are kept together in either train or test set
    """
    if split_field is None:
        # Simple random split
        train_indices = np.random.choice(
            range(len(dataset.ids)),
            size=int(len(dataset.ids) * (1 - test_size)),
            replace=False,
        )
        test_indices = np.random.permutation(
            np.setdiff1d(np.arange(len(dataset.ids)), train_indices)
        )
        train_indices = list(train_indices)
        test_indices = list(test_indices)
    else:
        # Split based on unique values of the field
        assert (
            split_field in dataset.other_fields
        ), f"Field {split_field} not found in dataset"
        unique_values = list(set(dataset.other_fields[split_field]))
        n_test = int(len(unique_values) * test_size)

        test_values = set(np.random.choice(unique_values, size=n_test, replace=False))

        train_indices = [
            i
            for i, val in enumerate(dataset.other_fields[split_field])
            if val not in test_values
        ]
        test_indices = [
            i
            for i, val in enumerate(dataset.other_fields[split_field])
            if val in test_values
        ]

    return dataset[train_indices], dataset[test_indices]


def create_cross_validation_splits(dataset: Dataset) -> list[Dataset]:
    pass


def cross_validate_probe(
    probe: LinearProbe, dataset_splits: list[Dataset]
) -> np.ndarray:
    accuracies = []

    for i, dataset in enumerate(dataset_splits):
        activations = get_activations(
            model=probe._llm,
            config=GenerateActivationsConfig(
                dataset=dataset,
                model_name=probe._llm.name,
                layer=probe.layer,
            ),
        )
        accuracy = compute_accuracy(probe, dataset, activations)
        accuracies.append(accuracy)

    return np.mean(np.array(accuracies), axis=1)


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

    train_dataset, test_dataset = create_train_test_split(dataset, split_field="index")

    print("TRAIN:", train_dataset.ids)
    print("TEST:", test_dataset.ids)

    train_config = GenerateActivationsConfig(
        dataset=train_dataset, model_name=model_name, layer=layer
    )
    test_config = GenerateActivationsConfig(
        dataset=test_dataset, model_name=model_name, layer=layer
    )

    # test_get_activations(
    #     config=train_config,
    #     model_name=model_name,
    # )
    test_compute_accuracy(
        model_name=model_name,
        train_config=train_config,
        test_config=test_config,
    )
