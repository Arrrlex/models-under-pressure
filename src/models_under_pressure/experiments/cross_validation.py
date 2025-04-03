"""
This script is used to choose the best layer for a given model and dataset.

It does this by training a probe on the train set and evaluating it on the test set.

It then repeats this process for each layer and reports the best layer.

"""

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
from tqdm import tqdm

from models_under_pressure.config import (
    LOCAL_MODELS,
    RESULTS_DIR,
    TRAIN_DIR,
    ChooseLayerConfig,
)
from models_under_pressure.interfaces.dataset import DatasetSpec, LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.interfaces.results import (
    CVFinalResults,
    CVIntermediateResults,
)
from models_under_pressure.probes.probes import ProbeFactory
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

    def splits(self) -> Iterator[Tuple[LabelledDataset, LabelledDataset]]:
        """Get train/test splits for cross validation.

        Returns:
            Sequence of (train, test) pairs where train is all folds except one
            and test is the held-out fold
        """
        for i in range(self.num_folds):
            # Test set is the current fold
            test = self.folds[i]

            # Train set is all other folds combined
            train_folds = self.folds[:i] + self.folds[i + 1 :]
            train = LabelledDataset.concatenate([fold for fold in train_folds])

            yield train, test


def _train_and_evaluate_fold(
    train: LabelledDataset,
    test: LabelledDataset,
    model_name: str,
    layer: int,
) -> float:
    """Worker function to train and evaluate a probe on a single fold.

    Args:
        train: Train dataset
        test: Test dataset
        model_name: Model name
        layer: Layer being evaluated

    Returns:
        Accuracy score for this fold
    """
    probe = ProbeFactory.build(
        probe_spec=ProbeSpec(
            name="sklearn_mean_agg_probe",
            hyperparams=None,
        ),
        model_name=model_name,
        layer=layer,
        train_dataset=train,
    )

    test_scores = probe.predict(test)
    return (np.array(test_scores) == test.labels_numpy()).mean()


def get_cross_validation_accuracies(
    model_name: str,
    layer: int,
    cv_splits: CVSplits,
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

    results = []
    for train, test in tqdm(
        cv_splits.splits(), total=cv_splits.num_folds, desc="Cross-validating"
    ):
        result = _train_and_evaluate_fold(
            train=train, test=test, model_name=model_name, layer=layer
        )
        results.append(result)

    return results


def choose_best_layer_via_cv(config: ChooseLayerConfig) -> CVFinalResults:
    """Main function to choose the best layer via cross validation.

    Args:
        config: ChooseLayerConfig
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    dataset = LabelledDataset.load_from(config.dataset_spec)

    train_dataset = dataset.filter(
        lambda x: x.other_fields.get("split", "train") == "train"
    )

    if config.max_samples is not None:
        train_dataset = train_dataset.sample(config.max_samples)

    results = CVIntermediateResults(config=config)

    cv_splits = CVSplits.create(train_dataset, config.cv_folds)

    print(f"Running cross-validation for {len(config.layers)} layers")
    for layer in print_progress(config.layers):
        print(f"Cross-validating layer {layer}...")
        layer_results = get_cross_validation_accuracies(
            model_name=config.model_name,
            layer=layer,
            cv_splits=cv_splits,
        )

        results.add_layer_results(layer, layer_results)
        results.save()

    print(f"Results: {results}")

    # Compute final results
    final_results = CVFinalResults.from_intermediate(results)

    # Save final results
    final_results.save()
    return final_results


if __name__ == "__main__":
    configs = [
        ChooseLayerConfig(
            model_name=LOCAL_MODELS[model_name],
            dataset_spec=DatasetSpec(
                path=TRAIN_DIR / "prompts_13_03_25_gpt-4o_filtered.jsonl",
                indices="all",
                field_mapping={},
                loader_kwargs={},
            ),
            max_samples=None,
            cv_folds=4,
            layers=list(range(0, max_layer, 2)),
            output_dir=RESULTS_DIR / "cross_validation",
        )
        for model_name, max_layer in [
            ("gemma-27b", 61),
            ("gemma-1b", 25),
            ("gemma-12b", 47),
        ]
    ]

    double_check_config(configs)

    for config in configs:
        choose_best_layer_via_cv(config)
