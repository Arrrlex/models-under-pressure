from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models_under_pressure.config import GENERATED_DATASET
from models_under_pressure.interfaces.dataset import LabelledDataset


@dataclass
class Variations:
    train_splits: dict[str, LabelledDataset]
    test_splits: dict[str, LabelledDataset]
    variation_values: list[str]


def split_by_variation(
    train_dataset: LabelledDataset,
    test_dataset: LabelledDataset,
    variation_type: str,
    max_samples: int | None = None,
) -> Variations:
    """Split the dataset into different splits for computing generalization heatmaps."""
    # Get unique values of variation_type
    variation_values = list(set(train_dataset.other_fields[variation_type]))
    test_variation_values = list(set(test_dataset.other_fields[variation_type]))
    if sorted(variation_values) != sorted(test_variation_values):
        print(f"Variation values: {variation_values}")
        print(f"Test variation values: {test_variation_values}")
    assert sorted(variation_values) == sorted(test_variation_values)

    train_splits = {}
    test_splits = {}
    for variation_value in variation_values:
        train_dataset_filtered = train_dataset.filter(
            lambda x: x.other_fields[variation_type] == variation_value
        )
        test_dataset_filtered = test_dataset.filter(
            lambda x: x.other_fields[variation_type] == variation_value
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

        train_splits[variation_value] = train_dataset_filtered
        test_splits[variation_value] = test_dataset_filtered

    return Variations(
        train_splits=train_splits,
        test_splits=test_splits,
        variation_values=variation_values,
    )


def create_cross_validation_splits(dataset: LabelledDataset) -> list[LabelledDataset]:
    raise NotImplementedError("Not implemented")


def load_train_test(
    dataset_path: Path,
) -> tuple[LabelledDataset, LabelledDataset]:
    """Load the train-test split for the generated dataset.

    Args:
        dataset_path: Path to the generated dataset

    Returns:
        tuple[LabelledDataset, LabelledDataset]: Train and test datasets
    """
    dataset = LabelledDataset.load_from(
        dataset_path,
        field_mapping=GENERATED_DATASET["field_mapping"],
    )

    train_dataset = dataset.filter(lambda x: x.other_fields["split"] == "train")
    test_dataset = dataset.filter(lambda x: x.other_fields["split"] == "test")

    return train_dataset, test_dataset


def load_filtered_train_dataset(
    dataset_path: Path,
    variation_type: str | None = None,
    variation_value: str | None = None,
    max_samples: int | None = None,
) -> LabelledDataset:
    # 1. Load train and eval datasets
    train_dataset, _ = load_train_test(dataset_path)

    # Filter for one variation type with specific value
    train_dataset = train_dataset.filter(
        lambda x: (
            variation_type is None or x.other_fields[variation_type] == variation_value
        )
    )

    # Subsample so this runs on the laptop
    if max_samples is not None:
        print("Subsampling the dataset ...")
        indices = np.random.choice(
            range(len(train_dataset.ids)),
            size=max_samples,
            replace=False,
        )
        train_dataset = train_dataset[list(indices)]  # type: ignore

    print(f"Number of samples in train dataset: {len(train_dataset.ids)}")
    return train_dataset
