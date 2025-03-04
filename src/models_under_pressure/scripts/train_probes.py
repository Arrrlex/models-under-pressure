import json
import os
from pathlib import Path

import dotenv
import numpy as np

from models_under_pressure.config import (
    ANTHROPIC_SAMPLES_CSV,
    GENERATED_DATASET_TRAIN_TEST_SPLIT,
    RESULTS_DIR,
    GenerateActivationsConfig,
)
from models_under_pressure.dataset.loaders import load_anthropic_csv, loaders
from models_under_pressure.interfaces.dataset import Dataset, Label
from models_under_pressure.interfaces.results import HeatmapResults
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

    return dataset[train_indices], dataset[test_indices]  # type: ignore


def create_generalization_variation_splits(
    train_dataset: Dataset,
    test_dataset: Dataset,
    variation_type: str,
    max_samples: int | None = None,
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
        train_dataset_filtered = train_dataset.filter(
            lambda x: x.other_fields["variation"] == variation_value
        )
        test_dataset_filtered = test_dataset.filter(
            lambda x: x.other_fields["variation"] == variation_value
        )

        if max_samples is not None:
            # Sample 80% for train, 20% for test
            train_size = int(max_samples * 0.8)
            test_size = int(max_samples * 0.2)

            train_indices = np.random.choice(
                range(len(train_dataset_filtered.ids)),
                size=min(train_size, len(train_dataset_filtered.ids)),
                replace=False,
            )
            test_indices = np.random.choice(
                range(len(test_dataset_filtered.ids)),
                size=min(test_size, len(test_dataset_filtered.ids)),
                replace=False,
            )

            train_dataset_filtered = train_dataset_filtered[list(train_indices)]  # type: ignore
            test_dataset_filtered = test_dataset_filtered[list(test_indices)]  # type: ignore

        train_datasets.append(train_dataset_filtered)
        test_datasets.append(test_dataset_filtered)

    return train_datasets, test_datasets, variation_values


def create_cross_validation_splits(dataset: Dataset) -> list[Dataset]:
    raise NotImplementedError("Not implemented")


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


def load_generated_dataset_split(
    dataset_path: Path,
    split_path: Path,
) -> tuple[Dataset, Dataset]:
    """Load the train-test split for the generated dataset.

    Args:
        dataset_path: Path to the generated dataset
        split_path: Path to save/load the train-test split

    Returns:
        tuple[Dataset, Dataset]: Train and test datasets
    """
    dataset = loaders["generated"](dataset_path)

    # Add a situations_ids field to the dataset (situations isn't hashable)
    dataset.other_fields["situations_ids"] = [  # type: ignore
        f"high_stakes_{s['high_stakes']}_low_stakes_{s['low_stakes']}"
        for s in dataset.other_fields["situations"]
    ]
    # TODO Check if there is no overlap between high and low stake situations

    if split_path.exists():
        split_dict = json.load(open(split_path))
        assert split_dict["dataset"] == dataset_path.stem

        train_indices = [
            dataset.ids.index(item_id) for item_id in split_dict["train_dataset"]
        ]
        test_indices = [
            dataset.ids.index(item_id) for item_id in split_dict["test_dataset"]
        ]
        train_dataset = dataset[train_indices]  # type: ignore
        test_dataset = dataset[test_indices]  # type: ignore
    else:
        # Create a train-test split with all data
        train_dataset, test_dataset = create_train_test_split(
            dataset, split_field="situations_ids"
        )
        split_dict = {
            "train_dataset": train_dataset.ids,
            "test_dataset": test_dataset.ids,
            "dataset": dataset_path.stem,
        }
        with open(split_path, "w") as f:
            json.dump(split_dict, f)

    return train_dataset, test_dataset


def generate_heatmap_for_generated_dataset(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_path: Path = Path("data/results/prompts_28_02_25.jsonl"),
    split_path: Path | None = None,
    variation_type: str = "prompt_style",
    layers: list[int] | None = None,
    max_samples: int | None = None,
) -> HeatmapResults:
    """Generate a heatmap for the generated dataset.

    This creates a global train-test split, then computes a heatmap for each layer
    by training on train set portions with a single variation value and evaluating
    on all test set portions with various variation values.

    Returns:
        HeatmapResults: Contains performances and variation values
    """
    if split_path is None:
        split_path = GENERATED_DATASET_TRAIN_TEST_SPLIT

    train_dataset, test_dataset = load_generated_dataset_split(
        dataset_path=dataset_path,
        split_path=split_path,
    )

    train_datasets, test_datasets, variation_values = (
        create_generalization_variation_splits(
            train_dataset, test_dataset, variation_type, max_samples=max_samples
        )
    )

    # Now to get the heat map, we train on each train dataset and evaluate on each test dataset
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
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
    return HeatmapResults(
        performances={
            layer: np.array(accuracies) for layer, accuracies in performances.items()
        },
        variation_values=variation_values,
        model_name=model_name,
        layers=layers,
        max_samples=max_samples,
    )


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    layers = [1, 10]
    max_samples = 20

    for variation_type in ["prompt_style", "tone", "language"]:
        print(f"\nGenerating heatmap for {variation_type}...")
        filename = RESULTS_DIR / f"generated_heatmap_{variation_type}.json"

        heatmap_results = generate_heatmap_for_generated_dataset(
            layers=layers,
            max_samples=max_samples,
            model_name=model_name,
            variation_type=variation_type,
        )
        print(heatmap_results.performances)
        print(heatmap_results.variation_values)

        json.dump(
            heatmap_results.to_dict(),
            open(filename, "w"),
        )
