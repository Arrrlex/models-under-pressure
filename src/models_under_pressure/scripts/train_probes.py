import os
from pathlib import Path

import dotenv
import numpy as np
import torch

from models_under_pressure.config import (
    ANTHROPIC_SAMPLES_CSV,
    BATCH_SIZE,
    GenerateActivationsConfig,
)
from models_under_pressure.dataset.loaders import load_anthropic_csv, loaders
from models_under_pressure.interfaces.dataset import Dataset, Label
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe, compute_accuracy

# Set random seed for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

dotenv.load_dotenv()


def get_activations(
    model: LLMModel,
    config: GenerateActivationsConfig,
    force_recompute: bool = False,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Get activations for a given model and config.

    Handle batching and caching of activations.
    """
    assert model.name == config.model_name

    if config.output_file.exists() and not force_recompute:
        return np.load(config.output_file)["activations"]
    else:
        print("Generating activations...")

        n_samples = len(config.dataset.inputs)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Get the shape from first batch to ensure consistency
        first_batch = config.dataset.inputs[0:1]
        first_activation = model.get_activations(
            inputs=first_batch, layers=[config.layer]
        )[0]
        activation_shape = first_activation.shape[1:]  # Remove batch dimension

        all_activations = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_inputs = config.dataset.inputs[start_idx:end_idx]

            print(f"Processing batch {i + 1}/{n_batches}...")
            batch_activations = model.get_activations(
                inputs=batch_inputs, layers=[config.layer]
            )[0]

            # Ensure all batches have the same shape by padding/truncating
            if batch_activations.shape[1:] != activation_shape:
                # Truncate or pad to match the first batch's shape
                padded_activations = np.zeros(
                    (batch_activations.shape[0],) + activation_shape
                )
                min_length = min(batch_activations.shape[1], activation_shape[0])
                padded_activations[:, :min_length] = batch_activations[:, :min_length]
                batch_activations = padded_activations

            all_activations.append(batch_activations)

        activations = np.concatenate(all_activations, axis=0)
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


def train_probes(
    dataset: Dataset, model_name: str, layers: list[int] | None = None
) -> dict[int, LinearProbe]:
    """Train a probe for each layer in the model."""
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )

    layers = layers or list(range(model.n_layers))

    if any(label == Label.AMBIGUOUS for label in dataset.labels):
        raise ValueError("Training dataset contains ambiguous labels")

    # Iterate over layers. For each layer, create a config, then train a probe and store it
    probes = {}
    for layer in layers:
        config = GenerateActivationsConfig(
            dataset=dataset,
            model_name=model_name,
            layer=layer,
        )
        print("Loading training activations...")
        activations = get_activations(
            model=model,
            config=config,
        )
        probe = LinearProbe(_llm=model, layer=layer)

        print("Training probe...")
        probe.fit(X=activations, y=config.dataset.labels_numpy())
        probes[layer] = probe
    return probes


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
        assert split_field in dataset.other_fields, (
            f"Field {split_field} not found in dataset"
        )
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


def create_generalization_variation_splits(
    train_dataset: Dataset,
    test_dataset: Dataset,
    variation_type: str,
) -> tuple[list[Dataset], list[Dataset], list[str]]:
    """Split the dataset into different splits for computing generalization heatmaps."""
    # Filter by variation_type
    train_dataset = train_dataset.filter(
        lambda x: x.other_fields["variation_type"] == variation_type
    )
    test_dataset = test_dataset.filter(
        lambda x: x.other_fields["variation_type"] == variation_type
    )

    if len(train_dataset.ids) == 0 or len(test_dataset.ids) == 0:
        print(f"Warning: No examples found for variation type {variation_type}")
        return [], [], []

    # Get unique values of variation_type
    variation_values = list(set(train_dataset.other_fields["variation"]))
    test_variation_values = list(set(test_dataset.other_fields["variation"]))
    assert sorted(variation_values) == sorted(test_variation_values)

    train_datasets = []
    test_datasets = []
    for variation_value in variation_values:
        train_datasets.append(
            train_dataset.filter(
                lambda x: x.other_fields["variation"] == variation_value
            )
        )
        test_datasets.append(
            test_dataset.filter(
                lambda x: x.other_fields["variation"] == variation_value
            )
        )

    return train_datasets, test_datasets, variation_values


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


def test_activations_on_anthropic_dataset():
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

    test_get_activations(
        config=train_config,
        model_name=model_name,
    )
    test_compute_accuracy(
        model_name=model_name,
        train_config=train_config,
        test_config=test_config,
    )


def generate_heatmap_for_generated_dataset(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: Path = Path("data/results/prompts_28_02_25.jsonl"),
    variation_type: str = "prompt_style",
    layers: list[int] | None = None,
    subsample_frac: float | None = None,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Generate a heatmap for the generated dataset.

    This creates a global train-test split, then computes a heatmap for each layer
    by training on train set portions with a single variation value and evaluating
    on all test set portions with various variation values.

    Args:
        model_name: Name of the model to use
        dataset_path: Path to the generated dataset
        layers: List of layers to use
        variation_type: Type of variation to use (prompt style, tone, language)
        subsample_frac: Fraction of the dataset to subsample

    Returns:
        dict[int, np.ndarray]: Layer index -> heatmap values (rows corresponding to indices of variation used for training)
        list[str]: Variation values
    """
    dataset = loaders["generated"](dataset_path)

    # Subsample so this runs on the laptop
    if subsample_frac is not None:
        print("Subsampling the dataset ...")
        indices = np.random.choice(
            range(len(dataset.ids)),
            size=int(len(dataset.ids) * subsample_frac),
            replace=False,
        )
        dataset = dataset[list(indices)]

    # Add a situations_ids field to the dataset (situations isn't hashable)
    dataset.other_fields["situations_ids"] = [
        f"high_stakes_{s['high_stakes']}_low_stakes_{s['low_stakes']}"
        for s in dataset.other_fields["situations"]
    ]

    train_dataset, test_dataset = create_train_test_split(
        dataset, split_field="situations_ids"
    )
    # TODO Store the split so we don't recompute
    # TODO Then split training set into training and validation so we don't optimize on test

    train_datasets, test_datasets, variation_values = (
        create_generalization_variation_splits(
            train_dataset, test_dataset, variation_type
        )
    )

    # Now to get the heat map, we train on each train dataset and evaluate on each test dataset
    model = LLMModel.load(
        model_name,
        model_kwargs={
            "token": os.getenv("HUGGINGFACE_TOKEN"),
            "device_map": "auto",
            "torch_dtype": torch.float16,
        },
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )
    layers = layers or list(range(model.n_layers))
    performances = {i: [] for i in layers}  # Layer index: heatmap values
    for i, train_ds in enumerate(train_datasets):
        print(f"Training on variation '{variation_type}'='{variation_values[i]}'")
        probes = train_probes(train_ds, model_name=model_name, layers=layers)

        for layer, probe in probes.items():
            accuracies = [
                compute_accuracy(
                    probe,
                    test_ds,
                    get_activations(
                        model,
                        GenerateActivationsConfig(
                            dataset=test_ds, model_name=model_name, layer=layer
                        ),
                    ),
                )
                for test_ds in test_datasets
            ]
            print(f"Layer {layer} accuracy: {accuracies}")
            performances[layer].append(accuracies)
    return {
        layer: np.array(accuracies) for layer, accuracies in performances.items()
    }, variation_values


if __name__ == "__main__":
    performances, variation_values = generate_heatmap_for_generated_dataset(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        layers=[1, 10],
        # subsample_frac=0.05,
    )
    print(performances)
    print(variation_values)
