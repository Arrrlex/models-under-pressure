import random
from pathlib import Path
from typing import Optional

import pandas as pd

from models_under_pressure.config import EVAL_DATASETS_BALANCED, TEST_DATASETS_BALANCED
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
                "input": sample.input,  # Keep track of the original input
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


def create_manual_review_excel_for_all_datasets(use_test: bool = True):
    if use_test:
        datasets = TEST_DATASETS_BALANCED
    else:
        datasets = EVAL_DATASETS_BALANCED

    output_path = Path(f"manual_review_{'test' if use_test else 'dev'}.xlsx")

    for dataset_name, dataset_path in datasets.items():
        if dataset_name == "manual" and use_test:
            continue

        print(f"Creating manual review for {dataset_name} ...")
        dataset = LabelledDataset.load_from(dataset_path)

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


def analyze_manual_review_results(
    excel_path: Path,
    dataset_name: str,
    use_test: bool = True,
) -> None:
    """Analyze manual review results from an Excel file.

    Args:
        excel_path: Path to the Excel file containing manual review results
        dataset_name: Name of the dataset being reviewed
        use_test: Whether to use test or dev datasets
    """
    # Load the original dataset to get ground truth labels
    if use_test:
        dataset_path = TEST_DATASETS_BALANCED[dataset_name]
    else:
        dataset_path = EVAL_DATASETS_BALANCED[dataset_name]
    original_dataset = LabelledDataset.load_from(dataset_path)
    ground_truth_labels = {
        record.id: record.label for record in original_dataset.to_records()
    }
    ground_truth_scale_labels = {
        record.id: int(record.other_fields["scale_labels"])
        for record in original_dataset.to_records()
    }

    # Load the Excel file
    excel_file = pd.ExcelFile(excel_path)

    # Get all sheets for this dataset
    dataset_sheets = [
        sheet
        for sheet in excel_file.sheet_names
        if str(sheet).startswith(f"{dataset_name}_")
    ]

    if not dataset_sheets:
        print(f"No review sheets found for dataset {dataset_name}")
        return

    # Load each reviewer's data
    reviewer_data = {}
    for sheet in dataset_sheets:
        reviewer_num = int(str(sheet).split("_")[-1])
        df = pd.read_excel(excel_file, sheet_name=sheet)

        # Convert scale labels to discrete labels using the same thresholds as in label_dataset
        scale_labels = df["label"].tolist()
        discrete_labels = []
        for score in scale_labels:
            if pd.isna(score):
                discrete_labels.append(None)
            else:
                try:
                    score_float = float(score)
                    if score_float <= 3:  # low_stakes_threshold
                        discrete_labels.append(Label.LOW_STAKES)
                    elif score_float >= 8:  # high_stakes_threshold
                        discrete_labels.append(Label.HIGH_STAKES)
                    else:
                        discrete_labels.append(Label.AMBIGUOUS)
                except (ValueError, TypeError):
                    discrete_labels.append(None)

        reviewer_data[reviewer_num] = {
            "ids": df["ids"].tolist(),
            "scale_labels": scale_labels,
            "discrete_labels": discrete_labels,
        }

    # Find items that have been annotated by at least two reviewers
    item_annotations = {}  # id -> list of reviewers who annotated it
    for reviewer_num, data in reviewer_data.items():
        for id_, scale_label in zip(data["ids"], data["scale_labels"]):
            if pd.notna(scale_label):  # Only count if there's an actual annotation
                if id_ not in item_annotations:
                    item_annotations[id_] = []
                item_annotations[id_].append(reviewer_num)

    # Only keep items with at least two annotations
    valid_items = {
        id_ for id_, reviewers in item_annotations.items() if len(reviewers) >= 2
    }
    print(f"\nFound {len(valid_items)} items with at least two annotations")

    # Calculate inter-rater agreement for both scale and discrete labels
    print(f"\nInter-rater agreement for {dataset_name}:")
    reviewer_pairs = [
        (r1, r2)
        for r1 in reviewer_data.keys()
        for r2 in reviewer_data.keys()
        if r1 < r2
    ]

    for r1, r2 in reviewer_pairs:
        # Get samples that both reviewers labeled and are in valid_items
        common_ids = (
            set(reviewer_data[r1]["ids"]) & set(reviewer_data[r2]["ids"]) & valid_items
        )

        # Calculate agreement for scale labels
        r1_scale = {
            id_: float(label)
            for id_, label in zip(
                reviewer_data[r1]["ids"], reviewer_data[r1]["scale_labels"]
            )
            if id_ in common_ids and pd.notna(label)
        }
        r2_scale = {
            id_: float(label)
            for id_, label in zip(
                reviewer_data[r2]["ids"], reviewer_data[r2]["scale_labels"]
            )
            if id_ in common_ids and pd.notna(label)
        }

        # Calculate agreement for discrete labels
        r1_discrete = {
            id_: label
            for id_, label in zip(
                reviewer_data[r1]["ids"], reviewer_data[r1]["discrete_labels"]
            )
            if id_ in common_ids and label is not None
        }
        r2_discrete = {
            id_: label
            for id_, label in zip(
                reviewer_data[r2]["ids"], reviewer_data[r2]["discrete_labels"]
            )
            if id_ in common_ids and label is not None
        }

        # Calculate scale agreement (using absolute difference)
        total_scale = len(r1_scale)
        if total_scale > 0:
            avg_diff = (
                sum(abs(r1_scale[id_] - r2_scale[id_]) for id_ in r1_scale)
                / total_scale
            )
            print(
                f"Reviewers {r1} and {r2} (scale): Average absolute difference = {avg_diff:.2f}"
            )

        # Calculate discrete agreement
        total_discrete = len(r1_discrete)
        if total_discrete > 0:
            matches = sum(
                1 for id_ in r1_discrete if r1_discrete[id_] == r2_discrete.get(id_)
            )
            agreement = matches / total_discrete * 100
            print(
                f"Reviewers {r1} and {r2} (discrete): {agreement:.1f}% agreement ({matches}/{total_discrete} samples)"
            )

    # Calculate agreement with ground truth
    print(f"\nAgreement with ground truth for {dataset_name}:")
    for reviewer_num, data in reviewer_data.items():
        # Get samples that were labeled by this reviewer, have ground truth, and are in valid_items
        valid_samples_scale = [
            (id_, float(label))
            for id_, label in zip(data["ids"], data["scale_labels"])
            if pd.notna(label) and id_ in ground_truth_labels and id_ in valid_items
        ]
        valid_samples_discrete = [
            (id_, label)
            for id_, label in zip(data["ids"], data["discrete_labels"])
            if label is not None and id_ in ground_truth_labels and id_ in valid_items
        ]

        # Calculate scale agreement with ground truth
        total_scale = len(valid_samples_scale)
        if total_scale > 0:
            avg_diff = (
                sum(
                    abs(score - ground_truth_scale_labels[id_])
                    for id_, score in valid_samples_scale
                )
                / total_scale
            )
            print(
                f"Reviewer {reviewer_num} (scale): Average absolute difference with ground truth = {avg_diff:.2f}"
            )

        # Calculate discrete agreement with ground truth
        total_discrete = len(valid_samples_discrete)
        if total_discrete > 0:
            matches = sum(
                1
                for id_, label in valid_samples_discrete
                if label == ground_truth_labels[id_]
            )
            agreement = matches / total_discrete * 100
            print(
                f"Reviewer {reviewer_num} (discrete): {agreement:.1f}% agreement with ground truth ({matches}/{total_discrete} samples)"
            )


if __name__ == "__main__":
    # create_manual_review_excel_for_all_datasets(use_test=True)
    datasets_and_files = {
        "~/Downloads/manual_review_test.xlsx": ["anthropic", "mt", "mts", "toolace"],
        "~/Downloads/manual_review_test_2.xlsx": [
            "redteaming",
            "mental_health",
            "mask",
        ],
    }
    for excel_path, dataset_names in datasets_and_files.items():
        for dataset_name in dataset_names:
            print(f"\nAnalyzing {dataset_name} ...".upper())
            analyze_manual_review_results(
                excel_path=Path(excel_path),
                dataset_name=dataset_name,
                use_test=True,
            )
