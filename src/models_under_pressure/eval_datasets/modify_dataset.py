from pathlib import Path
from typing import Any, List

from models_under_pressure.config import DATA_DIR, EVAL_DATASETS_RAW
from models_under_pressure.eval_datasets.label_dataset import LabelledDataset
from models_under_pressure.eval_datasets.mts_dataset import get_mts_samples_by_ids
from models_under_pressure.interfaces.dataset import Dataset


def combine_datasets(
    mts_dataset: LabelledDataset, samples_path: Path, output_path: Path
) -> LabelledDataset:
    """
    Combine fields from the MTS dataset with a samples dataset, using the samples dataset as the base.

    Args:
        mts_dataset: The original MTS dataset with labels
        samples_path: Path to the samples dataset file
        output_path: Path where to save the combined dataset

    Returns:
        The combined dataset
    """
    if not samples_path.exists():
        samples = get_mts_samples_by_ids(mts_dataset.ids)  # type: ignore
        samples.save_to(samples_path)
    else:
        samples = Dataset.load_from(samples_path)
    print(f"Loaded {len(samples.ids)} samples")

    # Create a mapping from id to index for both datasets
    dataset_id_to_idx = {id: idx for idx, id in enumerate(mts_dataset.ids)}
    samples_id_to_idx = {id: idx for idx, id in enumerate(samples.ids)}

    # Find common ids
    common_ids = set(mts_dataset.ids) & set(samples.ids)

    # Create new other_fields by combining fields from both datasets
    new_other_fields = {}

    # First add all fields from samples dataset
    for field_name, field_values in samples.other_fields.items():
        new_other_fields[field_name] = field_values

    # Then add fields from original dataset, maintaining order based on sample IDs
    for field_name, field_values in mts_dataset.other_fields.items():
        if field_name not in new_other_fields:
            # Create a new list for this field with the same length as samples
            new_values: List[Any] = [None] * len(samples.ids)  # type: ignore
            # Fill in values for common IDs
            for id in common_ids:
                sample_idx = samples_id_to_idx[id]
                dataset_idx = dataset_id_to_idx[id]
                new_values[sample_idx] = field_values[dataset_idx]
            new_other_fields[field_name] = new_values

    # Create a new LabelledDataset with combined fields
    combined_dataset = LabelledDataset(
        inputs=samples.inputs, ids=samples.ids, other_fields=new_other_fields
    )

    print(f"Combined dataset size: {len(combined_dataset)}")
    print(f"Original dataset size: {len(mts_dataset)}")
    print(f"Samples dataset size: {len(samples)}")
    print(f"Number of common IDs: {len(common_ids)}")

    # Save the combined dataset
    combined_dataset.save_to(output_path, overwrite=True)
    return combined_dataset


if __name__ == "__main__":
    # dataset = LabelledDataset.load_from(TEST_DATASETS_RAW["mts"])
    # samples_path = DATA_DIR / "temp/mts_test_updated.jsonl"
    # output_path = DATA_DIR / "temp/mts_test_combined.jsonl"
    dataset = LabelledDataset.load_from(EVAL_DATASETS_RAW["mts"])
    samples_path = DATA_DIR / "temp/mts_updated.jsonl"
    output_path = DATA_DIR / "temp/mts_combined.jsonl"

    # combine_datasets(dataset, samples_path, output_path)
    # print(Dataset.load_from(samples_path).ids)

    # dataset = LabelledDataset.load_from(output_path)
    # print(dataset.other_fields["labels"])
