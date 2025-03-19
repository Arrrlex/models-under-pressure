"""
This script is used to choose the best layer for a given model and dataset.

It does this by training a probe on the train set and evaluating it on the test set.

It then repeats this process for each layer and reports the best layer.

"""

import json
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from models_under_pressure.config import (
    LOCAL_MODELS,
    RESULTS_DIR,
    ChooseLayerConfig,
)
from models_under_pressure.experiments.dataset_splitting import load_train_test
from models_under_pressure.interfaces.activations import (
    Activation,
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe, compute_accuracy
from models_under_pressure.utils import double_check_config


class ChooseBestLayerResults(BaseModel):
    best_layer: int
    best_layer_accuracy: float
    layer_results: dict[int, list[float]]
    layer_mean_accuracies: dict[int, float]


@dataclass
class CVSplits:
    """
    A class that contains the cross validation splits for a given dataset.

    Note: we're not using scikit-learn's cross validation because we want to
    use the pair IDs to create the splits.
    """

    num_folds: int
    folds: list[LabelledDataset]

    @classmethod
    def create(cls, dataset: LabelledDataset, num_folds: int) -> "CVSplits":
        """Create cross validation splits from the dataset.

        Args:
            dataset: Dataset to split
            num_folds: Number of folds to create

        Returns:
            CVSplits object containing the folds
        """
        # Get unique pair IDs
        pair_ids = list(set(dataset.other_fields["pair_id"]))

        # Randomly shuffle pair IDs
        import numpy as np

        np.random.shuffle(pair_ids)

        # Split pair IDs into num_folds groups
        fold_size = len(pair_ids) // num_folds
        pair_id_folds = [
            pair_ids[i * fold_size : (i + 1) * fold_size] for i in range(num_folds)
        ]

        # Create dataset for each fold by filtering for its pair IDs
        folds = []
        for fold_pair_ids in pair_id_folds:
            fold_dataset = dataset.filter(
                lambda x: x.other_fields["pair_id"] in fold_pair_ids
            )
            folds.append(fold_dataset)

        return cls(num_folds=num_folds, folds=folds)


@dataclass
class DatasetWithActivations:
    """Simple wrapper class containing a dataset and its activations."""

    dataset: LabelledDataset
    activations: Activation


@dataclass
class CVSplitsWithActivations:
    """
    Wrapper class containing the cross validation splits and their activations.
    """

    cv_splits: CVSplits
    activation_folds: list[DatasetWithActivations]

    @classmethod
    def create(
        cls, cv_splits: CVSplits, llm: LLMModel, layer: int
    ) -> "CVSplitsWithActivations":
        """Create the CV splits with activations. Doing it this way is faster than getting the activations for each fold separately.

        Args:
            cv_splits: CVSplits
            llm: LLMModel
            layer: Layer to get activations for
        """
        # Get all activations at once
        combined_dataset = LabelledDataset.concatenate(cv_splits.folds)
        all_activations = llm.get_batched_activations(combined_dataset, layer)

        # Split activations according to fold lengths
        fold_lengths = [len(fold) for fold in cv_splits.folds]
        split_indices = [sum(fold_lengths[:i]) for i in range(1, len(fold_lengths))]
        activation_splits = all_activations.split(split_indices)

        # Create DatasetWithActivations for each fold
        activation_folds = [
            DatasetWithActivations(fold, act)
            for fold, act in zip(cv_splits.folds, activation_splits)
        ]
        return cls(cv_splits, activation_folds)

    def splits(self) -> Iterator[Tuple[DatasetWithActivations, DatasetWithActivations]]:
        """Get train/test splits for cross validation.

        Returns:
            Sequence of (train, test) pairs where train is all folds except one
            and test is the held-out fold
        """
        for i in range(self.cv_splits.num_folds):
            # Test set is the current fold
            test = self.activation_folds[i]

            # Train set is all other folds combined
            train_folds = self.activation_folds[:i] + self.activation_folds[i + 1 :]
            train = DatasetWithActivations(
                LabelledDataset.concatenate([fold.dataset for fold in train_folds]),
                Activation.concatenate([fold.activations for fold in train_folds]),
            )

            yield train, test


def get_cross_validation_accuracies(
    llm: LLMModel, layer: int, aggregator: Aggregator, cv_splits: CVSplits
) -> list[float]:
    """Get the cross validation accuracies for a given layer.

    Args:
        llm: LLMModel
        layer: Layer to evaluate
        aggregator: Aggregator
        cv_splits: CVSplits

    Returns:
        List of accuracies, one for each fold
    """
    results = []
    cv_splits_with_activations = CVSplitsWithActivations.create(cv_splits, llm, layer)
    for train, test in tqdm(
        cv_splits_with_activations.splits(),
        total=cv_splits.num_folds,
        desc="Cross-validating",
    ):
        probe = LinearProbe(_llm=llm, layer=layer, aggregator=aggregator)
        probe._fit(train.activations, train.dataset.labels_numpy())
        accuracy = compute_accuracy(probe, test.dataset, test.activations)
        results.append(accuracy)
    return results


def choose_best_layer_via_cv(config: ChooseLayerConfig) -> ChooseBestLayerResults:
    """Main function to choose the best layer via cross validation.

    Args:
        config: ChooseLayerConfig
    """
    train_dataset, _ = load_train_test(config.dataset_path)
    if config.max_samples is not None:
        train_dataset = train_dataset.sample(config.max_samples)

    cv_splits = CVSplits.create(train_dataset, config.cv_folds)

    llm = LLMModel.load(config.model_name)
    if config.layers is None:
        config.layers = list(range(llm.n_layers))
    else:
        assert all(0 <= layer < llm.n_layers for layer in config.layers)

    try:
        preprocessor = getattr(Preprocessors, config.preprocessor)
    except AttributeError:
        raise ValueError(f"Preprocessor {config.preprocessor} not found")
    try:
        postprocessor = getattr(Postprocessors, config.postprocessor)
    except AttributeError:
        raise ValueError(f"Postprocessor {config.postprocessor} not found")

    layer_accuracies = {}
    layer_mean_accuracies = {}
    for layer in config.layers:
        print(f"Cross-validating layer {layer}...")
        layer_accuracies[layer] = get_cross_validation_accuracies(
            llm=llm,
            layer=layer,
            aggregator=Aggregator(preprocessor, postprocessor),
            cv_splits=cv_splits,
        )
        layer_mean_accuracies[layer] = float(np.mean(layer_accuracies[layer]))

    # Find layer with highest mean accuracy
    best_layer = max(
        layer_mean_accuracies.keys(),
        key=lambda x: layer_mean_accuracies[x],
    )
    best_layer_accuracy = layer_mean_accuracies[best_layer]

    results = ChooseBestLayerResults(
        best_layer=best_layer,
        best_layer_accuracy=best_layer_accuracy,
        layer_results=layer_accuracies,
        layer_mean_accuracies=layer_mean_accuracies,
    )

    print("Results:")
    print(results)

    # Save results
    results_path = (
        RESULTS_DIR / "choose_best_layer_via_cross_validation" / config.output_filename
    )

    print(f"Saving results to {results_path}")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results.model_dump(), f)

    return results


if __name__ == "__main__":
    config = ChooseLayerConfig(
        model_name=LOCAL_MODELS["llama-1b"],
        dataset_path=RESULTS_DIR / "prompts_13_03_25_gpt-4o.jsonl",
        max_samples=100,
        cv_folds=5,
        preprocessor="mean",
        postprocessor="sigmoid",
    )
    double_check_config(config)

    choose_best_layer_via_cv(config)
