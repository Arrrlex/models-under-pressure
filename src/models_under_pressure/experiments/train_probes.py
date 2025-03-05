import json
from pathlib import Path

import dotenv
import numpy as np
import torch

from models_under_pressure.config import (
    BATCH_SIZE,
    RESULTS_DIR,
    HeatmapRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    create_cross_validation_splits,
    load_train_test,
    split_by_variation,
)
from models_under_pressure.interfaces.dataset import Dataset, Label, LabelledDataset
from models_under_pressure.interfaces.results import HeatmapResults
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe, compute_accuracy

# Set random seed for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

dotenv.load_dotenv()


def get_activations(
    model: LLMModel,
    dataset: Dataset,
    layer: int,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get activations for a given model and config.

    Handle batching and caching of activations.
    """

    print("Generating activations...")

    n_samples = len(dataset.inputs)
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Get the shape from first batch to ensure consistency
    first_batch = dataset.inputs[0:1]
    activations_tuple = model.get_activations(inputs=first_batch, layers=[layer])
    first_activation = activations_tuple[0][0]
    first_attn_mask = activations_tuple[1]
    activation_shape = first_activation.shape[1:]  # Remove batch dimension
    attn_mask_shape = first_attn_mask.shape[1:]  # Remove batch dimension

    all_activations = []
    all_attention_masks = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_inputs = dataset.inputs[start_idx:end_idx]

        activations_tuple = model.get_activations(inputs=batch_inputs, layers=[layer])
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
            padded_attn_mask = np.zeros((batch_attn_mask.shape[0],) + attn_mask_shape)
            min_length = min(batch_attn_mask.shape[1], attn_mask_shape[0])
            padded_attn_mask[:, :min_length] = batch_attn_mask[:, :min_length]
            batch_attn_mask = padded_attn_mask

        all_activations.append(batch_activations)
        all_attention_masks.append(batch_attn_mask)

    activations = np.concatenate(all_activations, axis=0)
    attention_mask = np.concatenate(all_attention_masks, axis=0)

    return activations, attention_mask


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
        activations, attention_mask = get_activations(
            model=model,
            dataset=dataset,
            layer=layer,
        )
        probe = LinearProbe(_llm=model, layer=layer)

        print("Training probe...")
        probe.fit(
            X=activations,
            y=dataset.labels_numpy(),
            attention_mask=attention_mask,
        )
        probes[layer] = probe
    return probes


def cross_validate_probe(
    probe: LinearProbe, dataset_splits: list[LabelledDataset]
) -> np.ndarray:
    accuracies = []

    for i, dataset in enumerate(dataset_splits):
        activations, attention_mask = get_activations(
            model=probe._llm,
            dataset=dataset,
            layer=probe.layer,
        )
        accuracy = compute_accuracy(
            probe, dataset, activations=activations, attention_mask=attention_mask
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


def generate_heatmap(
    config: HeatmapRunConfig,
    variation_type: str = "prompt_style",
) -> HeatmapResults:
    """Generate a heatmap for the generated dataset.

    This creates a global train-test split, then computes a heatmap for each layer
    by training on train set portions with a single variation value and evaluating
    on all test set portions with various variation values.

    Returns:
        HeatmapResults: Contains performances and variation values
    """
    train_dataset, test_dataset = load_train_test(
        dataset_path=config.dataset_path,
        split_path=config.split_path,
    )

    train_datasets, test_datasets, variation_values = split_by_variation(
        train_dataset, test_dataset, variation_type, max_samples=config.max_samples
    )

    # Now to get the heat map, we train on each train dataset and evaluate on each test dataset
    model = LLMModel.load(
        config.model_name,
        model_kwargs={"torch_dtype": torch.float16},
    )
    layers = config.layers
    performances = {i: [] for i in layers}  # Layer index: heatmap values
    for i, train_ds in enumerate(train_datasets):
        print(f"Training on variation '{variation_type}'='{variation_values[i]}'")
        probes = train_probes(model, train_ds, layers=config.layers)

        for layer, probe in probes.items():
            accuracies = []
            for test_ds in test_datasets:
                activations, attention_mask = get_activations(
                    model,
                    dataset=test_ds,
                    layer=layer,
                )
                accuracy = compute_accuracy(
                    probe,
                    test_ds,
                    activations=activations,
                    attention_mask=attention_mask,
                )
                accuracies.append(accuracy)
            performances[layer].append(accuracies)
    return HeatmapResults(
        performances={
            layer: np.array(accuracies) for layer, accuracies in performances.items()
        },
        variation_values=variation_values,
        variation_type=variation_type,
        model_name=config.model_name,
        layers=config.layers,
        max_samples=config.max_samples,
    )


if __name__ == "__main__":
    config = HeatmapRunConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        layers=[1, 10],
        max_samples=20,
        dataset_path=Path("data/results/prompts_28_02_25.jsonl"),
    )

    for variation_type in config.variation_types:
        print(f"\nGenerating heatmap for {variation_type}...")
        filename = RESULTS_DIR / f"generated_heatmap_{variation_type}.json"

        heatmap_results = generate_heatmap(
            config=config,
            variation_type=variation_type,
        )
        print(heatmap_results.performances)
        print(heatmap_results.variation_values)

        json.dump(
            heatmap_results.to_dict(),
            open(filename, "w"),
        )
