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


settings = [
    {
        "name": "anthropic_system_context",
        "dataset": "anthropic",
        "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
        "labelling_method": "scale",
    },
    {
        "name": "anthropic_system_and_labelling_context",
        "dataset": "anthropic",
        "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
        "labelling_method": "anthropic_context",
    },
    {
        "name": "anthropic_system_and_complex_labelling_context",
        "dataset": "anthropic",
        "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
        "labelling_method": "anthropic_extended_context",
    },
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
]


def create_labelling_variations(
    *,
    out_dir: Path,
    labelling_model: str,
    max_samples: int | None = None,
):
    for setting in settings:
        print(f"Labelling {setting['dataset']} with {setting['name']}")
        dataset = LabelledDataset.load_from(EVAL_DATASETS[setting["dataset"]])
        labelled_dataset = label_dataset(
            dataset=dataset.sample(max_samples),  # type: ignore
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
    for setting in settings:
        if setting["dataset"] == dataset_name:
            filename = f"{setting['name']}_labelled"
            if max_samples is not None:
                filename += f"_{max_samples}"
            filename += ".jsonl"
            file_path = variations_dir / filename
            if file_path.exists():
                dataset_files.append((setting["name"], file_path))

    if not dataset_files:
        raise ValueError(f"No labelled variations found for dataset {dataset_name}")

    # Load the first dataset to get the base structure
    first_name, first_file = dataset_files[0]
    base_dataset = LabelledDataset.load_from(first_file)

    # Create a new dictionary for other_fields
    combined_other_fields = {}

    # Process each variation
    for name, file_path in dataset_files:
        variation_dataset = LabelledDataset.load_from(file_path)

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
            if field_name != "labels":  # Skip the main labels field
                combined_other_fields[f"{prefix}{field_name}"] = field_values

        # Add the labels field with the appropriate prefix
        if "labels" in variation_dataset.other_fields:
            combined_other_fields[f"{prefix}labels"] = variation_dataset.other_fields[
                "labels"
            ]

    # Create the combined dataset with the new other_fields dictionary
    combined_dataset = Dataset(
        inputs=base_dataset.inputs,
        ids=base_dataset.ids,
        other_fields=combined_other_fields,
    )

    return combined_dataset


if __name__ == "__main__":
    max_samples = 10
    out_dir = DATA_DIR / "results" / "labelling_comparison"

    create_labelling_variations(
        out_dir=out_dir,
        labelling_model="gpt-4o-mini",
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

    # The combined dataset can then be viewed with the dashboard
