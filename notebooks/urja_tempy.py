from pathlib import Path
from typing import Any

from models_under_pressure.config import (
    RunConfig,
)
from models_under_pressure.eval_datasets.label_dataset import (
    LabelledDataset,
)


def label_and_filter():
    # Define the input file path
    config = RunConfig(run_id="debug")
    # input_file = Path(f"{config.run_dir}/prompts_13_03_25_gpt-4o.jsonl")
    # output_file = Path(f"{config.run_dir}/prompts_{config.suffix}_labeled.jsonl")
    # field_mapping = {
    #     "prompt": "inputs",
    #     "id": "ids",
    # }
    # # Load the dataset
    # print(f"Loading dataset from {input_file}")
    # dataset = Dataset.load_from(input_file, field_mapping=field_mapping)

    # # Label the dataset
    # labeled_dataset = label_dataset(  # type: ignore
    #     dataset=dataset,
    #     model="gpt-4o",
    #     max_concurrent=10,
    #     use_rubric=False,
    #     force_override=False,
    # )

    # # Save the labeled dataset
    # print(f"Saving labeled dataset to {output_file}")
    # labeled_dataset.save_to(output_file)

    # print("Labeling complete!")

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
    filtered_output_file = Path(
        f"{config.run_dir}/prompts_13_03_25_gpt-4o_filtered.jsonl"
    )

    labelled_dataset = LabelledDataset.load_from(
        Path(f"{config.run_dir}/prompts_13_03_25_gpt-4o_labeled.jsonl")
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


# for dataset_name in EVAL_DATASETS_BALANCED.keys():
# for dataset_name in ["synthetic"]:
#     input_file = Path(SYNTHETIC_DATASET_PATH)
#     output_file = Path(str(SYNTHETIC_DATASET_PATH).replace(".csv", "_relabeled.jsonl"))
# try:
#     output_file = Path(
#         str(EVAL_DATASETS_BALANCED[dataset_name]).replace(
#             ".jsonl", "_relabeled.jsonl"
#         )
#     )
# except Exception:
#     output_file = Path(
#         str(EVAL_DATASETS_BALANCED[dataset_name]).replace(
#             ".csv", "_relabeled.jsonl"
#         )
#     )

# field_mapping = {
#     "text": "inputs",
#     "id": "ids",
# }

# print(f"Loading dataset {dataset_name} from {input_file}")
# dataset = Dataset.load_from(input_file, field_mapping=field_mapping)

# # Label the dataset
# labeled_dataset = label_dataset(
#     dataset=dataset,
#     model="gpt-4o",
#     max_concurrent=50,
#     use_rubric=False,
#     force_override=False,
# )

# # Save the labeled dataset
# print(f"Saving labeled dataset to {output_file}")
# labeled_dataset.save_to(output_file, overwrite=True)
# print(f"Labeling complete for {dataset_name}!")

#
