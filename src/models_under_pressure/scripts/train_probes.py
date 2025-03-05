import json
import os
from pathlib import Path

import dotenv
import numpy as np
import torch

from models_under_pressure.config import (
    BATCH_SIZE,
    EVAL_DATASETS,
    RESULTS_DIR,
    GenerateActivationsConfig,
    HeatmapRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    create_cross_validation_splits,
    create_train_test_split,
    load_train_test,
    split_by_variation,
)
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
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
    cache: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get activations for a given model and config.

    Handle batching and caching of activations.
    """
    assert model.name == config.model_name

    if cache and (
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

        if cache:
            np.savez_compressed(config.acts_output_file, activations=activations)
            np.savez_compressed(
                config.attn_mask_output_file, attention_mask=attention_mask
            )
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
    dataset: LabelledDataset, model_name: str, layers: list[int] | None = None
) -> dict[int, LinearProbe]:
    """Train a probe for each layer in the model."""
    model = LLMModel.load(model_name)

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


def cross_validate_probe(
    probe: LinearProbe, dataset_splits: list[LabelledDataset]
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


def cross_validate_probes(
    probes: list[LinearProbe], dataset: LabelledDataset
) -> np.ndarray:
    dataset_splits = create_cross_validation_splits(dataset)
    accuracies = np.array(
        [cross_validate_probe(probe, dataset_splits) for probe in probes]
    )
    return np.mean(accuracies, axis=1)


def test_activations_on_anthropic_dataset():
    layer = 10
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print("Loading dataset...")
    dataset = LabelledDataset.load_from(
        EVAL_DATASETS["anthropic"]["path"],
        field_mapping=EVAL_DATASETS["anthropic"]["field_mapping"],
    )

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
    # TODO The model is read here but will also be initialized inside train_probes, which seems inefficient
    # TODO Get rid of unneeded arguments (LLMModel.load(model_name) is used in train_probes and seems to work fine)
    model = LLMModel.load(
        config.model_name,
        model_kwargs={
            "token": os.getenv("HUGGINGFACE_TOKEN"),
            "device_map": "auto",
            "torch_dtype": torch.float16,
        },
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )
    layers = config.layers
    performances = {i: [] for i in layers}  # Layer index: heatmap values
    for i, train_ds in enumerate(train_datasets):
        print(f"Training on variation '{variation_type}'='{variation_values[i]}'")
        probes = train_probes(
            train_ds, model_name=config.model_name, layers=config.layers
        )

        for layer, probe in probes.items():
            accuracies = []
            for test_ds in test_datasets:
                activations, attention_mask = get_activations(
                    model,
                    GenerateActivationsConfig(
                        dataset=test_ds, model_name=config.model_name, layer=layer
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
