from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from models_under_pressure.activation_store import ActivationStore
from models_under_pressure.interfaces.dataset import (
    Dataset,
    LabelledDataset,
    subsample_balanced_subset,
)
from models_under_pressure.model import LLMModel


def create_train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    split_field: str | None = None,
) -> tuple[LabelledDataset, LabelledDataset]:
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
        assert (
            split_field in dataset.other_fields
        ), f"Field {split_field} not found in dataset"
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


def load_dataset(
    dataset_path: Path,
    dataset_filters: dict[str, Any] | None = None,
    model_name: str | None = None,
    layer: int | None = None,
    compute_activations: bool = False,
    variation_type: str | None = None,
    variation_value: str | None = None,
    n_per_class: int | None = None,
    mmap: bool = True,
) -> LabelledDataset:
    """Load the dataset.

    If model_name and layer are provided, the activations are loaded and added to the dataset.
    If compute_activations is True, the activations are computed; if False,
    the activations are loaded from the activation store.

    If variation_type and variation_value are provided, the dataset is filtered to only
    include examples where variation_type has the given variation value.

    If n_per_class is provided, the dataset is subsampled to contain only n_per_class
    examples per class.

    Args:
        dataset_path: Path to the dataset
        model_name: Name of the model to load activations for
        layer: Layer to load activations for
        compute_activations: Whether to compute activations
        variation_type: Type of variation to filter on
        variation_value: Value of variation to filter on
        n_per_class: Number of samples per class to load
    """
    dataset = LabelledDataset.load_from(dataset_path)
    if model_name is not None and layer is not None and not compute_activations:
        dataset = ActivationStore().enrich(
            dataset,
            path=dataset_path,
            model_name=model_name,
            layer=layer,
        )

    if variation_type is not None and variation_value is not None:
        dataset = dataset.filter(
            lambda x: x.other_fields[variation_type] == variation_value
        )

    dataset = LabelledDataset.load_from(dataset_path)

    if model_name is not None and layer is not None and not compute_activations:
        dataset = ActivationStore().enrich(
            dataset,
            path=dataset_path,
            model_name=model_name,
            layer=layer,
            mmap=mmap,
        )

    if variation_type is not None and variation_value is not None:
        dataset = dataset.filter(
            lambda x: x.other_fields[variation_type] == variation_value
        )
    if dataset_filters is not None:
        dataset = dataset.filter(
            lambda x: all(
                x.other_fields[key] == value for key, value in dataset_filters.items()
            )
        )

    if n_per_class is not None and len(dataset) > n_per_class * 2:
        dataset = subsample_balanced_subset(dataset, n_per_class=n_per_class)

    if model_name is not None and layer is not None and compute_activations:
        model = LLMModel.load(model_name)
        activations = model.get_batched_activations(dataset, layer=layer)
        dataset = dataset.assign(
            activations=activations.activations,
            attention_mask=activations.attention_mask,
            input_ids=activations.input_ids,
        )

    return dataset


class LazyLoader:
    def __init__(
        self,
        loader: Callable[..., LabelledDataset],
        kwargs: dict[str, dict[str, Any]],
        **common_kwargs: Any,
    ):
        self.loader = loader
        self.kwargs = kwargs
        self.common_kwargs = common_kwargs

    def __getitem__(self, key: str) -> LabelledDataset:
        kwargs = self.kwargs[key] | self.common_kwargs
        return self.loader(**kwargs)


def get_split(dataset: LabelledDataset, split_name: str) -> LabelledDataset:
    return dataset.filter(lambda x: x.split == split_name)


def load_splits_lazy(
    dataset_path: Path,
    dataset_filters: dict[str, Any] | None = None,
    model_name: str | None = None,
    layer: int | None = None,
    compute_activations: bool = False,
    variation_type: str | None = None,
    variation_value: str | None = None,
    n_per_class: int | None = None,
    mmap: bool = True,
) -> LazyLoader:
    common_kwargs = {
        "dataset_filters": dataset_filters,
        "model_name": model_name,
        "layer": layer,
        "compute_activations": compute_activations,
        "variation_type": variation_type,
        "variation_value": variation_value,
        "n_per_class": n_per_class,
        "mmap": mmap,
    }

    if dataset_path.is_dir():
        split_names = [p.stem for p in dataset_path.glob("*.jsonl")]
        loader_kwargs = {
            split_name: {"dataset_path": dataset_path / f"{split_name}.jsonl"}
            for split_name in split_names
        }
        return LazyLoader(loader=load_dataset, kwargs=loader_kwargs, **common_kwargs)
    else:
        dataset = load_dataset(dataset_path, **common_kwargs)

        if "split" in dataset.other_fields:
            kwargs = {
                str(s): {"dataset": dataset, "split_name": str(s)}
                for s in set(dataset.other_fields["split"])
            }
            return LazyLoader(loader=get_split, kwargs=kwargs)
        else:
            return LazyLoader(
                loader=lambda x: x, kwargs={"train": {"dataset": dataset}}
            )


def load_train_test(
    dataset_path: Path,
    model_name: str | None = None,
    layer: int | None = None,
    compute_activations: bool = False,
    variation_type: str | None = None,
    variation_value: str | None = None,
    n_per_class: int | None = None,
    mmap: bool = True,
) -> tuple[LabelledDataset, LabelledDataset]:
    splits = load_splits_lazy(
        dataset_path,
        model_name=model_name,
        layer=layer,
        compute_activations=compute_activations,
        variation_type=variation_type,
        variation_value=variation_value,
        n_per_class=n_per_class,
        mmap=mmap,
    )
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    return train_dataset, test_dataset
