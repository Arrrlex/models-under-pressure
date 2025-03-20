# Make sure that there is no overlap between the dev and test datasets

from models_under_pressure.config import (
    EVAL_DATASETS_RAW,
    TEST_DATASETS_RAW,
    TEST_EVALS_DIR,
)
from models_under_pressure.interfaces.dataset import (
    Dataset,
    LabelledDataset,
    subsample_balanced_subset,
)


def check_overlap(dataset_name: str):
    dev_dataset = Dataset.load_from(EVAL_DATASETS_RAW[dataset_name])
    test_dataset = Dataset.load_from(TEST_DATASETS_RAW[dataset_name])

    overlap = {}

    # Check overlap in ids
    id_overlap = set(dev_dataset.ids) & set(test_dataset.ids)
    if id_overlap:
        print(
            f"ID Overlap found in {dataset_name} for {len(id_overlap)} samples: {id_overlap}"
        )
    else:
        print(f"No ID overlap found in {dataset_name}")
    overlap["ids"] = id_overlap

    # Check whether any test dataset inputs are present in the dev dataset
    input_overlap = []
    for ix, input in enumerate(test_dataset.inputs):
        for ex, dev_input in enumerate(dev_dataset.inputs):
            case = {
                "test_id": test_dataset.ids[ix],
                "dev_id": dev_dataset.ids[ex],
                "test_input": input,
                "dev_input": dev_input,
            }

            # Handle both string inputs and dialogue inputs
            if isinstance(input, str) and isinstance(dev_input, str):
                if input in dev_input:
                    input_overlap.append(case)
            else:
                # For dialogue inputs, compare the messages
                if len(input) == len(dev_input):
                    messages_match = all(
                        test_msg.role == dev_msg.role  # type: ignore
                        and test_msg.content in dev_msg.content  # type: ignore
                        for test_msg, dev_msg in zip(input, dev_input)
                    )
                    if messages_match:
                        input_overlap.append(case)

    if input_overlap:
        print(
            f"Input overlap found in {dataset_name} for {len(input_overlap)} samples"  #: {input_overlap}"
        )
    else:
        print(f"No input overlap found in {dataset_name}")

    overlap["input"] = input_overlap

    # Now get non-overlapping ids
    non_overlapping_ids = set(test_dataset.ids) - set(dev_dataset.ids)
    for case in overlap["input"]:
        if case["test_id"] in non_overlapping_ids:
            non_overlapping_ids.remove(case["test_id"])

    return overlap, non_overlapping_ids


if __name__ == "__main__":
    # for dataset_name in EVAL_DATASETS_RAW.keys():
    for dataset_name in ["mt"]:
        print(f"\nChecking overlap for {dataset_name}")
        overlap, good_ids = check_overlap(dataset_name)

        if len(overlap["input"]) > 0 or len(overlap["ids"]) > 0:
            dataset = LabelledDataset.load_from(TEST_DATASETS_RAW[dataset_name])

            print(f"Before filtering: {len(dataset)}")
            dataset = dataset.filter(lambda x: x.id in good_ids)
            print(f"After filtering: {len(dataset)}")

            cleaned_output_path = TEST_EVALS_DIR / f"{dataset_name}_clean.jsonl"
            dataset.save_to(cleaned_output_path, overwrite=True)

            print("Subsampling the data to get a balanced dataset")
            dataset = subsample_balanced_subset(dataset)

            # Save the balanced data
            balanced_output_path = (
                TEST_EVALS_DIR / f"{dataset_name}_clean_balanced.jsonl"
            )
            print(f"Saving the balanced data to {balanced_output_path}")
            dataset.save_to(balanced_output_path, overwrite=True)
