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
    SYNTHETIC_DATASET_PATH,
    ChooseLayerConfig,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.interfaces.results import (
    CVFinalResults,
    CVIntermediateResults,
)
from models_under_pressure.model import LLMModel
from models_under_pressure.probes.probe_factory import ProbeFactory
from models_under_pressure.utils import batched_range, double_check_config


@dataclass
class CVSplits:
    """
    A class that contains the cross validation splits for a given dataset.

    Note: we're not using scikit-learn's cross validation because we want to
    use the pair IDs to create the splits.
    """

    num_folds: int
    folds: list[list[int]]

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

        # Create indices for each fold by finding indices matching pair IDs
        folds = []
        for fold_pair_ids in pair_id_folds:
            fold_indices = [
                i
                for i, x in enumerate(dataset.other_fields["pair_id"])
                if x in fold_pair_ids
            ]
            folds.append(fold_indices)

        return cls(num_folds=num_folds, folds=folds)

    def splits(
        self, dataset: LabelledDataset
    ) -> Iterator[Tuple[LabelledDataset, LabelledDataset]]:
        """Get train/test splits for cross validation.

        Returns:
            Sequence of (train, test) pairs where train is all folds except one
            and test is the held-out fold
        """
        for i in range(self.num_folds):
            # Train indices are all indices except the current fold
            train_indices = [
                idx for fold in self.folds[:i] + self.folds[i + 1 :] for idx in fold
            ]
            # Test set is the current fold
            test_indices = self.folds[i]

            yield dataset[train_indices], dataset[test_indices]


def get_cross_validation_accuracies(
    dataset: LabelledDataset,
    cv_splits: CVSplits,
    probe_spec: ProbeSpec,
) -> list[float]:
    """Get the cross validation accuracies for a given layer.

    Args:
        dataset: Dataset to evaluate
        cv_splits: CVSplits
        probe_spec: ProbeSpec

    Returns:
        List of accuracies, one for each fold
    """
    results = []

    for train, test in cv_splits.splits(dataset):
        probe = ProbeFactory.build(probe_spec=probe_spec, train_dataset=train)
        test_scores = probe.predict(test)
        results.append((np.array(test_scores) == test.labels_numpy()).mean())

    return results


def choose_best_layer_via_cv(config: ChooseLayerConfig) -> CVFinalResults:
    """Main function to choose the best layer via cross validation."""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    dataset = LabelledDataset.load_from(config.dataset_path)
    dataset = dataset.filter(lambda x: x.other_fields.get("split", "train") == "train")

    if config.max_samples is not None:
        dataset = dataset.sample(config.max_samples)

    llm = LLMModel.load(config.model_name)

    if config.layers is None:
        layers = list(range(llm.n_layers))
    else:
        assert all(0 <= layer < llm.n_layers for layer in config.layers)
        layers = config.layers

    results = CVIntermediateResults(config=config)

    cv_splits = CVSplits.create(dataset, config.cv_folds)

    pbar = tqdm(total=len(layers), desc="Cross-validating layers")

    for batch_start, batch_end in batched_range(len(layers), config.layer_batch_size):
        batch_layers = layers[batch_start:batch_end]
        activations, inputs = llm.get_batched_activations_for_layers(
            dataset,
            layers=batch_layers,
            batch_size=config.batch_size,
        )

        for layer_idx, layer in enumerate(batch_layers):
            dataset = dataset.assign(
                activations=activations[layer_idx],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            layer_results = get_cross_validation_accuracies(
                dataset=dataset,
                cv_splits=cv_splits,
                probe_spec=config.probe_spec,
            )

            results.add_layer_results(layer, layer_results)
            results.save()

            pbar.update(1)

    pbar.close()
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
            dataset_path=SYNTHETIC_DATASET_PATH,
            max_samples=None,
            cv_folds=4,
            layers=list(range(0, max_layer, 2)),
            batch_size=4,
            output_dir=RESULTS_DIR / "cross_validation",
            probe_spec=ProbeSpec(name="sklearn_mean_agg_probe"),
        )
        for model_name, max_layer in [
            ("gemma-27b", 61),
            ("gemma-1b", 25),
            ("gemma-12b", 47),
            ("llama-1b", 16),
        ]
    ]

    double_check_config(configs)

    for config in configs:
        choose_best_layer_via_cv(config)
