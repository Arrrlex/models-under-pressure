from pathlib import Path
from typing import Any

from models_under_pressure.config import DATA_DIR, TEST_EVALS_DIR
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import (
    Dataset,
    LabelledDataset,
)


def label_and_filter(dataset_name: str):
    # Define the input file path
    field_mapping = {}
    if dataset_name == "mentalhealth":
        input_file = Path(f"{DATA_DIR}/combined_data.csv")
        output_file = Path(f"{TEST_EVALS_DIR}/mentalhealth_labeled.jsonl")
        field_mapping = {
            "statement": "inputs",
            "id": "ids",
        }
    elif dataset_name == "redteaming":
        input_file = Path(f"{DATA_DIR}/aya_redteaming_dataset.csv")
        output_file = Path(f"{TEST_EVALS_DIR}/aya_redteaming_labeled.jsonl")
        field_mapping = {
            "prompt": "inputs",
            "id": "ids",
        }
    # Load the dataset
    print(f"Loading dataset from {input_file}")
    dataset = Dataset.load_from(input_file, field_mapping=field_mapping)

    # Label the dataset
    labeled_dataset = label_dataset(  # type: ignore
        dataset=dataset,
        model="gpt-4o",
        max_concurrent=10,
        use_rubric=False,
        force_override=False,
    )

    # Save the labeled dataset
    print(f"Saving labeled dataset to {output_file}")
    labeled_dataset.save_to(output_file)

    print("Labeling complete!")

    def filter_function(record1: Any, record2: Any) -> bool:
        # Check if the record has the required fields
        if not hasattr(record1, "other_fields") or not hasattr(record2, "other_fields"):
            return False

        # Get the label and confidence
        label1 = record1.other_fields.get("labels")
        # situation_id1 = record1.other_fields.get("situations")[0]
        situation_id1 = record1.other_fields.get("situations")["high_stakes"]

        label2 = record2.other_fields.get("labels")
        situation_id2 = record2.other_fields.get("situations")["high_stakes"]

        # Check if situation IDs match
        if not situation_id1 or not situation_id2 or situation_id1 != situation_id2:
            return False

        # Check if either record has ambiguous label or insufficient confidence
        if (
            label1 not in ["high-stakes", "low-stakes"]
            or label2
            not in [
                "high-stakes",
                "low-stakes",
            ]
            or label1 == label2
        ):
            return False

        return True

    # Define the output file path
    filtered_output_file = Path(f"{TEST_EVALS_DIR}/{dataset_name}_filtered.jsonl")

    labelled_dataset = LabelledDataset.load_from(
        Path(f"{TEST_EVALS_DIR}/{dataset_name}_labeled.jsonl")
    )
    filtered_records = []
    records = labelled_dataset.to_records()
    for i in range(0, len(records) - 1, 2):
        record1 = records[i]
        record2 = records[i + 1]
        if filter_function(record1, record2):
            filtered_records.extend([record1, record2])
    filtered_dataset = LabelledDataset.from_records(filtered_records)

    # Save the filtered dataset
    print(f"Saving filtered dataset to {filtered_output_file}")
    filtered_dataset.save_to(filtered_output_file, overwrite=True)

    # Print statistics
    total_records = len(labelled_dataset)
    filtered_records = len(filtered_dataset)
    print(f"Total records: {total_records}")
    print(f"Records after filtering: {filtered_records}")
    print(f"Removed {total_records - filtered_records} records")

    # Print label distribution of filtered dataset
    if isinstance(filtered_dataset, LabelledDataset):
        filtered_dataset.print_label_distribution()

    print("Processing complete!")


# label_and_filter()

# Install datasets library if you haven't already
# pip install datasets


# Load the dataset
# dataset = load_dataset("CohereForAI/aya_redteaming")
# json.dump(dataset, open("datasets/aya_redteaming_dataset.json", "w"))
