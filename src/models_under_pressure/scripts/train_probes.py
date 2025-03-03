import json
import os
from pathlib import Path

import dotenv
import numpy as np
import torch

from models_under_pressure.config import (
    ANTHROPIC_SAMPLES_CSV,
    BATCH_SIZE,
    GENERATED_DATASET_TRAIN_TEST_SPLIT,
    RESULTS_DIR,
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get activations for a given model and config.

    Handle batching and caching of activations.
    """
    assert model.name == config.model_name

    if (
        config.acts_output_file.exists()
        and config.attn_mask_output_file.exists()
        and not force_recompute
    ):
        return (
            np.load(config.acts_output_file)["activations"],
            np.load(config.attn_mask_output_file)["attention_mask"],
        )
    else:
        print("Generating activations...")

        n_samples = len(config.dataset.inputs)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Get the shape from first batch to ensure consistency
        first_batch = config.dataset.inputs[0:1]
        activations_tuple = model.get_activations(
            inputs=first_batch, layers=[config.layer]
        )
        first_activation = activations_tuple[0][0]
        first_attn_mask = activations_tuple[1]
        activation_shape = first_activation.shape[1:]  # Remove batch dimension
        attn_mask_shape = first_attn_mask.shape[1:]  # Remove batch dimension

        all_activations = []
        all_attention_masks = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_inputs = config.dataset.inputs[start_idx:end_idx]

            activations_tuple = model.get_activations(
                inputs=batch_inputs, layers=[config.layer]
            )
            batch_activations = activations_tuple[0][0]
            batch_attn_mask = activations_tuple[1]

            # Ensure all batches have the same shape by padding/truncating
            if batch_activations.shape[1:] != activation_shape:
                padded_activations = np.zeros(
                    (batch_activations.shape[0],) + activation_shape
                )
                min_length = min(batch_activations.shape[1], activation_shape[0])
                padded_activations[:, :min_length] = batch_activations[:, :min_length]
                batch_activations = padded_activations

            if batch_attn_mask.shape[1:] != attn_mask_shape:
                padded_attn_mask = np.zeros(
                    (batch_attn_mask.shape[0],) + attn_mask_shape
                )
                min_length = min(batch_attn_mask.shape[1], attn_mask_shape[0])
                padded_attn_mask[:, :min_length] = batch_attn_mask[:, :min_length]
                batch_attn_mask = padded_attn_mask

            all_activations.append(batch_activations)
            all_attention_masks.append(batch_attn_mask)

        activations = np.concatenate(all_activations, axis=0)
        attention_mask = np.concatenate(all_attention_masks, axis=0)

        np.savez_compressed(config.acts_output_file, activations=activations)
        np.savez_compressed(config.attn_mask_output_file, attention_mask=attention_mask)
        return activations, attention_mask


def test_get_activations(config: GenerateActivationsConfig, model_name: str):
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )
    activations, attention_mask = get_activations(
        model=model, config=config, force_recompute=True
    )
    # Load precomputed activations
    activations2, attention_mask2 = get_activations(
        model=model,
        config=config,
        force_recompute=False,
    )
    assert np.allclose(activations, activations2)
    assert np.allclose(attention_mask, attention_mask2)


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
    activations, attention_mask = get_activations(
        model=model,
        config=train_config,
    )
    if any(label == Label.AMBIGUOUS for label in train_config.dataset.labels):
        raise ValueError("Training dataset contains ambiguous labels")

    probe = LinearProbe(_llm=model, layer=train_config.layer)

    print("Training probe...")
    probe.fit(
        X=activations,
        y=train_config.dataset.labels_numpy(),
        attention_mask=attention_mask,
    )

    print("Loading testing activations...")
    activations, attention_mask = get_activations(
        model=model,
        config=test_config,
    )
    if any(label == Label.AMBIGUOUS for label in test_config.dataset.labels):
        raise ValueError("Test dataset contains ambiguous labels")

    print("Computing accuracy...")
    accuracy = compute_accuracy(
        probe,
        test_config.dataset,
        activations=activations,
        attention_mask=attention_mask,
    )
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
        activations, attention_mask = get_activations(
            model=model,
            config=config,
        )
        probe = LinearProbe(_llm=model, layer=layer)

        print("Training probe...")
        probe.fit(
            X=activations,
            y=config.dataset.labels_numpy(),
            attention_mask=attention_mask,
        )
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

    return dataset[train_indices], dataset[test_indices]  # type: ignore


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
    raise NotImplementedError("Not implemented")


def cross_validate_probe(
    probe: LinearProbe, dataset_splits: list[Dataset]
) -> np.ndarray:
    accuracies = []

    for i, dataset in enumerate(dataset_splits):
        activations, attention_mask = get_activations(
            model=probe._llm,
            config=GenerateActivationsConfig(
                dataset=dataset,
                model_name=probe._llm.name,
                layer=probe.layer,
            ),
        )
        accuracy = compute_accuracy(
            probe, dataset, activations=activations, attention_mask=attention_mask
        )
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
    split_path: Path | None = None,
    variation_type: str = "prompt_style",
    layers: list[int] | None = None,
    subsample_frac: float | None = None,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Generate a heatmap for the generated dataset.

    This creates a global train-test split, then computes a heatmap for each layer
    by training on train set portions with a single variation value and evaluating
    on all test set portions with various variation values.

    Returns:
        dict[int, np.ndarray]: Layer index -> heatmap values (rows corresponding to indices of variation used for training)
        list[str]: Variation values
    """
    if split_path is None:
        split_path = GENERATED_DATASET_TRAIN_TEST_SPLIT
        if subsample_frac is not None:
            # Insert subsample fraction before file extension
            stem = split_path.stem
            suffix = split_path.suffix
            split_path = split_path.with_name(
                f"{stem}_subsample_{subsample_frac}{suffix}"
            )

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
        # Subsample so this runs on the laptop
        if subsample_frac is not None:
            print("Subsampling the dataset ...")
            indices = np.random.choice(
                range(len(dataset.ids)),
                size=int(len(dataset.ids) * subsample_frac),
                replace=False,
            )
            dataset = dataset[list(indices)]  # type: ignore
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
            accuracies = []
            for test_ds in test_datasets:
                activations, attention_mask = get_activations(
                    model,
                    GenerateActivationsConfig(
                        dataset=test_ds, model_name=model_name, layer=layer
                    ),
                )
                accuracy = compute_accuracy(
                    probe,
                    test_ds,
                    activations=activations,
                    attention_mask=attention_mask,
                )
                accuracies.append(accuracy)
            performances[layer].append(accuracies)
    return {
        layer: np.array(accuracies) for layer, accuracies in performances.items()
    }, variation_values


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    layers = [1, 10, 15]
    subsample_frac = None

    for variation_type in ["prompt_style", "tone", "language"]:
        print(f"\nGenerating heatmap for {variation_type}...")
        filename = RESULTS_DIR / f"generated_heatmap_{variation_type}.json"

        performances, variation_values = generate_heatmap_for_generated_dataset(
            layers=layers,
            subsample_frac=subsample_frac,
            model_name=model_name,
            variation_type=variation_type,
        )
        print(performances)
        print(variation_values)

        json.dump(
            {
                "performances": {
                    layer: performances[layer].tolist() for layer in layers
                },
                "variation_values": variation_values,
                "model_name": model_name,
                "layers": layers,
                "subsample_frac": subsample_frac,
            },
            open(filename, "w"),
        )
