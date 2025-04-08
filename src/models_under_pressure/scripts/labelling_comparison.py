from pathlib import Path

from models_under_pressure.config import DATA_DIR, EVAL_DATASETS
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import (
    Dataset,
    LabelledDataset,
    Message,
    Record,
)

anthropic_system_prompt = "You are an LLM as part of a general purpose chatbot."
mt_system_prompt = "You are an LLM as part of a medical assistant system, interacting with medical practitioners to improve efficiency."


def add_system_prompt(x: Record, system_prompt: str) -> Record:
    if isinstance(x.input, str):
        x.input = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=x.input),
        ]
    else:  # Dialogue type
        x.input = [Message(role="system", content=system_prompt)] + x.input  # type: ignore
    return x


settings = {
    "anthropic": [
        {
            "name": "anthropic_system_context",
            "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
            "labelling_method": "scale",
        },
        {
            "name": "anthropic_system_and_labelling_context",
            "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
            "labelling_method": "anthropic_context",
        },
        {
            "name": "anthropic_system_and_complex_labelling_context",
            "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
            "labelling_method": "anthropic_extended_context",
        },
    ],
    "mt": [
        {
            "name": "mt_system_context",
            "dataset": "mt",
            "preprocessing_fn": lambda x: add_system_prompt(x, mt_system_prompt),
            "labelling_method": "scale",
        },
        {
            "name": "mt_system_and_labelling_context",
            "dataset": "mt",
            "preprocessing_fn": lambda x: add_system_prompt(x, mt_system_prompt),
            "labelling_method": "mt_context",
        },
        {
            "name": "mt_system_and_complex_labelling_context",
            "dataset": "mt",
            "preprocessing_fn": lambda x: add_system_prompt(x, mt_system_prompt),
            "labelling_method": "mt_extended_context",
        },
    ],
}


def create_labelling_variations(
    *,
    out_dir: Path,
    labelling_model: str,
    max_samples: int | None = None,
):
    for dataset_name, dataset_settings in settings.items():
        dataset = LabelledDataset.load_from(EVAL_DATASETS[dataset_name])
        if max_samples is not None:
            dataset = dataset.sample(max_samples)

        for setting in dataset_settings:
            print(f"Labelling {dataset_name} with {setting['name']}")
            labelled_dataset = label_dataset(
                dataset=dataset,  # type: ignore
                model=labelling_model,
                preprocessing_fn=setting["preprocessing_fn"],
                labelling_method=setting["labelling_method"],
            )
            filename = f"{setting['name']}_labelled"
            if max_samples is not None:
                filename += f"_{max_samples}"
            filename += ".jsonl"
            labelled_dataset.save_to(
                out_dir / filename,
                overwrite=True,
            )


def combine_labelling_variations(
    *,
    dataset_name: str,
    variations_dir: Path,
    max_samples: int | None = None,
) -> Dataset:
    """
    Read all the labelling variations for a dataset and combine them into a single LabelledDataset.

    Args:
        dataset_name: Name of the dataset (e.g., "anthropic" or "mt")
        variations_dir: Directory containing the labelled variations
        max_samples: Optional maximum number of samples to include

    Returns:
        A combined LabelledDataset with prefixed columns for each variation
    """
    # Find all relevant files for this dataset
    dataset_files = []
    for setting in settings[dataset_name]:
        filename = f"{setting['name']}_labelled"
        if max_samples is not None:
            filename += f"_{max_samples}"
        filename += ".jsonl"
        file_path = variations_dir / filename
        if file_path.exists():
            dataset_files.append((setting["name"], file_path))

    if not dataset_files:
        raise ValueError(f"No labelled variations found for dataset {dataset_name}")

    # Load the first variation dataset to get the sample IDs
    first_variation_path = dataset_files[0][1]
    first_variation = LabelledDataset.load_from(first_variation_path)
    sample_ids = set(first_variation.ids)

    # Load the base dataset and filter it to match the labelled datasets
    base_dataset = LabelledDataset.load_from(EVAL_DATASETS[dataset_name])

    # Filter the base dataset to only include samples with IDs in sample_ids
    filtered_indices = [i for i, id in enumerate(base_dataset.ids) if id in sample_ids]
    if len(filtered_indices) != len(sample_ids):
        raise ValueError(
            f"Not all sample IDs from the labelled dataset were found in the base dataset. "
            f"Expected {len(sample_ids)} samples, found {len(filtered_indices)}."
        )

    # Create filtered base dataset
    filtered_base_dataset = Dataset(
        inputs=[base_dataset.inputs[i] for i in filtered_indices],
        ids=[base_dataset.ids[i] for i in filtered_indices],
        other_fields={
            field: [values[i] for i in filtered_indices]
            for field, values in base_dataset.other_fields.items()
        },
    )

    # Create a new dictionary for other_fields
    combined_other_fields = {}

    # Include the original labels from the filtered base dataset without a prefix
    for field in filtered_base_dataset.other_fields:
        combined_other_fields[field] = filtered_base_dataset.other_fields[field]

    # Process each variation
    for name, file_path in dataset_files:
        variation_dataset = LabelledDataset.load_from(file_path)

        # Assert that all variation datasets have the same sample IDs
        variation_ids = set(variation_dataset.ids)
        if variation_ids != sample_ids:
            raise ValueError(
                f"Sample IDs mismatch in {name}. Expected {len(sample_ids)} samples, got {len(variation_ids)}. "
                f"First variation has {len(sample_ids)} samples."
            )

        # Determine the prefix based on the variation name
        if "system_context" in name:
            prefix = "system_"
        elif "system_and_labelling_context" in name:
            prefix = "system_label_"
        elif "system_and_complex_labelling_context" in name:
            prefix = "system_label_plus_"
        else:
            prefix = f"{name}_"

        # Add all fields from the variation with the appropriate prefix
        for field_name, field_values in variation_dataset.other_fields.items():
            if "label" in field_name:  # Skip the main labels field
                combined_other_fields[f"{prefix}{field_name}"] = field_values

    # Create the combined dataset with the new other_fields dictionary
    combined_dataset = Dataset(
        inputs=filtered_base_dataset.inputs,
        ids=filtered_base_dataset.ids,
        other_fields=combined_other_fields,
    )

    return combined_dataset


def find_label_disagreements(
    *,
    combined_dataset: Dataset,
    dataset_name: str,
) -> Dataset:
    """
    Filter the combined dataset to find cases with disagreements between different labelling methods.

    Args:
        combined_dataset: The combined dataset with all labelling variations
        dataset_name: Name of the dataset (e.g., "anthropic" or "mt")

    Returns:
        A filtered dataset containing only records with disagreements
    """
    # Get all label fields from the dataset
    label_fields = [
        field
        for field in combined_dataset.other_fields.keys()
        if field.endswith("labels") and "scale_label" not in field
    ]

    # If there are no label fields, return an empty dataset
    if not label_fields:
        return Dataset(inputs=[], ids=[], other_fields={})

    # Create a list to store indices of records with disagreements
    disagreement_indices = []

    # Check each record for disagreements
    for i in range(len(combined_dataset)):
        # Get all labels for this record
        record_labels = {}
        for field in label_fields:
            if field in combined_dataset.other_fields:
                label_value = combined_dataset.other_fields[field][i]
                # Convert to Label enum if it's an integer
                if isinstance(label_value, int):
                    from models_under_pressure.interfaces.dataset import Label

                    label_value = Label.from_int(label_value)
                record_labels[field] = label_value

        # Check if there are any disagreements
        if len(set(record_labels.values())) > 1:
            disagreement_indices.append(i)

    # Create a filtered dataset with only the records with disagreements
    if disagreement_indices:
        # Use the filter method instead of direct indexing
        filtered_dataset = combined_dataset.filter(
            lambda r: combined_dataset.ids.index(r.id) in disagreement_indices
        )
        print(
            f"Found {len(disagreement_indices)} records with disagreements in {dataset_name} dataset"
        )
        return filtered_dataset
    else:
        print(f"No disagreements found in {dataset_name} dataset")
        return Dataset(inputs=[], ids=[], other_fields={})


if __name__ == "__main__":
    max_samples = 100
    out_dir = DATA_DIR / "results" / "labelling_comparison"

    create_labelling_variations(
        out_dir=out_dir,
        labelling_model="gpt-4o",
        max_samples=max_samples,
    )

    for dataset_name in ["anthropic", "mt"]:
        combined_dataset = combine_labelling_variations(
            dataset_name=dataset_name,
            variations_dir=out_dir,
            max_samples=max_samples,
        )
        combined_dataset.save_to(
            out_dir / f"{dataset_name}_combined_labelled.jsonl",
            overwrite=True,
        )

        # Find and save disagreements
        disagreements = find_label_disagreements(
            combined_dataset=combined_dataset,
            dataset_name=dataset_name,
        )
        if len(disagreements) > 0:
            disagreements.save_to(
                out_dir / f"{dataset_name}_disagreements.jsonl",
                overwrite=True,
            )

    # The combined dataset can then be viewed with the dashboard
