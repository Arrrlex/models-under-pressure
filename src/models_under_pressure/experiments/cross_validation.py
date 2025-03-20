"""
This script is used to choose the best layer for a given model and dataset.

It does this by training a probe on the train set and evaluating it on the test set.

It then repeats this process for each layer and reports the best layer.

"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Iterator, Tuple

import numpy as np
from tqdm import tqdm

from models_under_pressure.config import (
    LOCAL_MODELS,
    RESULTS_DIR,
    TRAIN_DIR,
    ChooseLayerConfig,
)
from models_under_pressure.interfaces.activations import (
    Activation,
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe, compute_accuracy
from models_under_pressure.utils import double_check_config, print_progress


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
        cls, cv_splits: CVSplits, llm: LLMModel, layer: int, batch_size: int
    ) -> "CVSplitsWithActivations":
        """Create the CV splits with activations. Doing it this way is faster than getting the activations for each fold separately.

        Args:
            cv_splits: CVSplits
            llm: LLMModel
            layer: Layer to get activations for
        """
        # Get all activations at once
        combined_dataset = LabelledDataset.concatenate(cv_splits.folds)
        all_activations = llm.get_batched_activations(
            combined_dataset, layer=layer, batch_size=batch_size
        )

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


def _train_and_evaluate_fold(
    train_test_pair: Tuple[DatasetWithActivations, DatasetWithActivations],
    layer: int,
    aggregator: Aggregator,
) -> float:
    """Worker function to train and evaluate a probe on a single fold.

    Args:
        train_test_pair: Tuple of (train, test) DatasetWithActivations
        llm: LLMModel
        layer: Layer being evaluated
        aggregator: Aggregator for the probe

    Returns:
        Accuracy score for this fold
    """
    train, test = train_test_pair
    probe = LinearProbe(_llm=None, layer=layer, aggregator=aggregator)  # type: ignore
    probe._fit(train.activations, train.dataset.labels_numpy())
    return compute_accuracy(probe, test.dataset, test.activations)


def get_cross_validation_accuracies(
    llm: LLMModel,
    layer: int,
    aggregator: Aggregator,
    cv_splits: CVSplits,
    batch_size: int,
) -> list[float]:
    """Get the cross validation accuracies for a given layer.

    Args:
        llm: LLMModel
        layer: Layer to evaluate
        aggregator: Aggregator
        cv_splits: CVSplits
        batch_size: Batch size for processing

    Returns:
        List of accuracies, one for each fold
    """
    cv_splits_with_activations = CVSplitsWithActivations.create(
        cv_splits, llm, layer, batch_size
    )

    # Create list of train/test pairs
    fold_pairs = list(cv_splits_with_activations.splits())

    # Create partial function with fixed arguments
    worker_fn = partial(_train_and_evaluate_fold, layer=layer, aggregator=aggregator)

    # Use multiprocessing to evaluate folds in parallel
    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap(worker_fn, fold_pairs),
                total=cv_splits.num_folds,
                desc="Cross-validating",
            )
        )

    return results


def main(config: ChooseLayerConfig):
    """Main function to choose the best layer via cross validation.

    Args:
        config: ChooseLayerConfig
    """
    if config.output_path.exists():
        raise FileExistsError(f"Results already exist for {config.output_path}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    dataset = LabelledDataset.load_from(**config.dataset_spec)
    if "split" in dataset.other_fields:
        train_dataset = dataset.filter(lambda x: x.other_fields["split"] == "train")
    else:
        train_dataset = dataset
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

    results = {
        "config": config.model_dump(),
        "date": datetime.now().isoformat(),
        "layer_results": {},
        "layer_mean_accuracies": {},
    }

    print(f"Running cross-validation for {len(config.layers)} layers")
    for layer in print_progress(config.layers):
        print(f"Cross-validating layer {layer}...")
        results["layer_results"][layer] = get_cross_validation_accuracies(
            llm=llm,
            layer=layer,
            aggregator=Aggregator(preprocessor, postprocessor),
            cv_splits=cv_splits,
            batch_size=config.batch_size,
        )
        results["layer_mean_accuracies"][layer] = float(
            np.mean(results["layer_results"][layer])
        )

        with open(config.temp_output_path, "a") as f:
            f.write(json.dumps(results) + "\n")

    # Find layer with highest mean accuracy
    results["best_layer"] = max(
        results["layer_mean_accuracies"].keys(),
        key=lambda x: results["layer_mean_accuracies"][x],
    )
    results["best_layer_accuracy"] = results["layer_mean_accuracies"][
        results["best_layer"]
    ]
    print("Results:")
    print(results)

    # Save results

    if config.max_samples is not None:
        print("Not saving results, because we sampled a subset of the dataset")
    else:
        print(f"Saving results to {config.output_path}")

        with open(config.output_path, "a") as f:
            f.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    config = ChooseLayerConfig(
        model_name=LOCAL_MODELS["llama-70b"],
        dataset_spec={
            "file_path_or_name": TRAIN_DIR / "manual_upsampled.csv",
            "field_mapping": {"id": "ids"},
        },
        max_samples=None,
        cv_folds=4,
        preprocessor="mean",
        postprocessor="sigmoid",
        layers=list(range(10, 40, 2)),
        batch_size=16,
        output_dir=RESULTS_DIR / "cross_validation",
    )
    double_check_config(config)

    main(config)
