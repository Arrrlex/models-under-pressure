import json

import numpy as np

from models_under_pressure.config import LOCAL_MODELS, RESULTS_DIR, HeatmapRunConfig
from models_under_pressure.experiments.dataset_splitting import (
    load_train_test,
    split_by_variation,
)
from models_under_pressure.experiments.train_probes import train_probes
from models_under_pressure.interfaces.results import HeatmapResults
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import compute_accuracy


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
    model = LLMModel.load(config.model_name)
    layers = config.layers
    performances = {i: [] for i in layers}  # Layer index: heatmap values
    for i, train_ds in enumerate(train_datasets):
        print(f"Training on variation '{variation_type}'='{variation_values[i]}'")
        probes = train_probes(model, train_ds, layers=config.layers)

        for layer, probe in probes.items():
            accuracies = []
            for test_ds in test_datasets:
                activations_obj = probe._llm.get_batched_activations(
                    dataset=test_ds,
                    layer=layer,
                )
                accuracy = compute_accuracy(
                    probe,
                    test_ds,
                    activations_obj,
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
    best_layers = {}
    best_layer_accuracies = {}

    config = HeatmapRunConfig(
        layers=list(range(1, 12)),
        max_samples=None,
        model_name=LOCAL_MODELS["llama-8b"],
    )

    output_dir = RESULTS_DIR / "generate_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    for variation_type in config.variation_types:
        print(f"\nGenerating heatmap for {variation_type}...")
        out_path = output_dir / config.output_filename(variation_type)

        heatmap_results = generate_heatmap(
            config=config,
            variation_type=variation_type,
        )
        print(heatmap_results.performances)
        print(heatmap_results.variation_values)

        json.dump(
            heatmap_results.to_dict(),
            open(out_path, "w"),
        )
        # Calculate mean accuracy across all variations for each layer
        layer_means = {
            layer: np.mean(accuracies)
            for layer, accuracies in heatmap_results.performances.items()
        }

        # Find best performing layer
        best_layers[variation_type] = max(layer_means.items(), key=lambda x: x[1])[0]
        best_layer_accuracies[variation_type] = layer_means[best_layers[variation_type]]

    for variation_type, best_layer in best_layers.items():
        print(
            f"Best layer for {variation_type}: {best_layer} (mean accuracy: {best_layer_accuracies[variation_type]:.3f})"
        )
