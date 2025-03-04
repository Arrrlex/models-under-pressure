import json
import os
from pathlib import Path

import dotenv
import numpy as np

from models_under_pressure.config import (
    ANTHROPIC_SAMPLES_CSV,
    RESULTS_DIR,
    GenerateActivationsConfig,
    HeatmapRunConfig,
)
from models_under_pressure.dataset.loaders import load_anthropic_csv
from models_under_pressure.experiments.dataset_splitting import (
    create_cross_validation_splits,
    create_generalization_variation_splits,
    create_train_test_split,
    load_generated_dataset_split,
)
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
    train_dataset, test_dataset = load_generated_dataset_split(
        dataset_path=config.dataset_path,
        split_path=config.split_path,
    )

    train_datasets, test_datasets, variation_values = (
        create_generalization_variation_splits(
            train_dataset, test_dataset, variation_type, max_samples=config.max_samples
        )
    )

    # Now to get the heat map, we train on each train dataset and evaluate on each test dataset
    model = LLMModel.load(
        config.model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )
    layers = config.layers or list(range(model.n_layers))
    performances = {i: [] for i in layers}  # Layer index: heatmap values
    for i, train_ds in enumerate(train_datasets):
        print(f"Training on variation '{variation_type}'='{variation_values[i]}'")
        probes = train_probes(
            train_ds, model_name=config.model_name, layers=config.layers
        )

        for layer, probe in probes.items():
            accuracies = [
                compute_accuracy(
                    probe,
                    test_ds,
                    get_activations(
                        model,
                        GenerateActivationsConfig(
                            dataset=test_ds, model_name=config.model_name, layer=layer
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

        heatmap_results = generate_heatmap_for_generated_dataset(
            config=config,
            variation_type=variation_type,
        )
        print(heatmap_results.performances)
        print(heatmap_results.variation_values)

        json.dump(
            heatmap_results.to_dict(),
            open(filename, "w"),
        )
