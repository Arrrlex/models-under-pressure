import random
from pathlib import Path
from typing import Optional

import pandas as pd

from models_under_pressure.config import EVAL_DATASETS_BALANCED
from models_under_pressure.interfaces.dataset import Label, LabelledDataset


def create_manual_review_excel(
    dataset: LabelledDataset,
    dataset_name: str,
    num_reviewers: int,
    num_samples: int,
    output_path: Path,
    seed: Optional[int] = None,
    append: bool = False,
) -> None:
    """Create an Excel sheet for manual review with multiple tabs.

    Args:
        dataset: The dataset to sample from
        dataset_name: Name of the dataset
        num_reviewers: Number of reviewers (tabs) to create
        num_samples: Number of samples per reviewer
        output_path: Path to save the Excel file
        seed: Random seed for reproducibility
        append: Whether to append to an existing Excel file
    """
    if seed is not None:
        random.seed(seed)

    # Split data into high and low stakes
    high_stakes_samples = []
    low_stakes_samples = []

    for i, label in enumerate(dataset.labels):
        if label == Label.HIGH_STAKES:
            high_stakes_samples.append(i)
        elif label == Label.LOW_STAKES:
            low_stakes_samples.append(i)

    # Calculate samples per label per reviewer
    samples_per_label = num_samples // 2

    # Pre-select samples for each label
    high_stakes_selected = random.sample(
        high_stakes_samples,
        k=samples_per_label * num_reviewers // 2,  # Each sample appears twice
    )

    low_stakes_selected = random.sample(
        low_stakes_samples,
        k=samples_per_label * num_reviewers // 2,  # Each sample appears twice
    )

    reviewer_dfs = []
    # For each reviewer
    for reviewer in range(num_reviewers):
        # Calculate the sliding window indices
        # Each window overlaps by half with the previous and next window
        window_size = samples_per_label // 2
        start_idx = reviewer * window_size
        end_idx = start_idx + samples_per_label

        # Get high stakes samples for this reviewer
        if end_idx > len(high_stakes_selected):
            # Take remaining samples from end and wrap around to beginning
            remaining = end_idx - len(high_stakes_selected)
            reviewer_high_stakes = (
                high_stakes_selected[start_idx:] + high_stakes_selected[:remaining]
            )
        else:
            reviewer_high_stakes = high_stakes_selected[start_idx:end_idx]

        # Get low stakes samples for this reviewer
        if end_idx > len(low_stakes_selected):
            # Take remaining samples from end and wrap around to beginning
            remaining = end_idx - len(low_stakes_selected)
            reviewer_low_stakes = (
                low_stakes_selected[start_idx:] + low_stakes_selected[:remaining]
            )
        else:
            reviewer_low_stakes = low_stakes_selected[start_idx:end_idx]

        # Combine samples
        reviewer_samples = reviewer_high_stakes + reviewer_low_stakes

        # Create DataFrame for this reviewer
        data = []
        for sample_idx in reviewer_samples:
            sample = dataset[sample_idx]
            if isinstance(sample.input, str):
                system_message = ""
                conversation = sample.input
            else:
                system_message = next(
                    (msg.content for msg in sample.input if msg.role == "system"),
                    "",
                )
                conversation = "\n\n".join(
                    f"{msg.role.upper()}\n{msg.content}"
                    for msg in sample.input
                    if msg.role != "system"
                )

            data_dict = {
                "ids": sample.id,
                "system_message": system_message,
                "conversation": conversation,
                "label": "",
                "comments": "",
            }
            if dataset_name == "toolace":
                data_dict["original_system_prompt"] = sample.other_fields[
                    "original_system_prompts"
                ]
            data.append(data_dict)

        random.shuffle(data)

        reviewer_df = pd.DataFrame(data)
        reviewer_dfs.append(reviewer_df)

    # Shuffle the reviewer dataframes so that different people are paired up for different datasets
    random.shuffle(reviewer_dfs)

    # Create a writer object with append mode if specified
    mode = "a" if append else "w"
    with pd.ExcelWriter(output_path, engine="openpyxl", mode=mode) as writer:
        for reviewer_df, reviewer in zip(reviewer_dfs, range(num_reviewers)):
            # Write to Excel sheet with dataset name in sheet name
            sheet_name = f"{dataset_name}_{reviewer + 1}"
            reviewer_df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    # dataset_name = "anthropic"
    dataset_names = EVAL_DATASETS_BALANCED.keys()
    output_path = Path("manual_review.xlsx")

    for dataset_name in dataset_names:
        print(f"Creating manual review for {dataset_name} ...")
        dataset = LabelledDataset.load_from(EVAL_DATASETS_BALANCED[dataset_name])

        # Create the Excel sheet
        create_manual_review_excel(
            dataset=dataset,
            dataset_name=dataset_name,
            num_reviewers=4,
            num_samples=20,  # 10 high stakes + 10 low stakes per reviewer
            output_path=output_path,
            seed=42,  # For reproducibility
            append=True if output_path.exists() else False,
        )
