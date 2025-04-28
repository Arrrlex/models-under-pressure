import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

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


def compute_agreement_metrics(
    reviewer_data: Dict[int, Dict[str, List]],
    ground_truth_labels: Dict[str, Label],
    ground_truth_scale_labels: Dict[str, int],
    valid_items: set,
) -> Tuple[
    float,
    Tuple[float, float],
    float,
    Tuple[float, float],
    float,
    Tuple[float, float],
    float,
    Tuple[float, float],
]:
    """Compute agreement metrics between reviewers and with ground truth.

    Returns:
        Tuple containing:
        - Mean absolute difference between reviewers (scale)
        - Confidence interval for scale difference between reviewers
        - Mean absolute difference with ground truth (scale)
        - Confidence interval for scale difference with ground truth
        - Agreement rate between reviewers (discrete)
        - Confidence interval for discrete agreement between reviewers
        - Agreement rate with ground truth (discrete)
        - Confidence interval for discrete agreement with ground truth
    """
    # Get all reviewer pairs
    reviewer_pairs = [
        (r1, r2)
        for r1 in reviewer_data.keys()
        for r2 in reviewer_data.keys()
        if r1 < r2
    ]

    # Compute scale differences between reviewers
    scale_diffs = []
    for r1, r2 in reviewer_pairs:
        common_ids = (
            set(reviewer_data[r1]["ids"]) & set(reviewer_data[r2]["ids"]) & valid_items
        )
        for id_ in common_ids:
            idx1 = reviewer_data[r1]["ids"].index(id_)
            idx2 = reviewer_data[r2]["ids"].index(id_)
            if pd.notna(reviewer_data[r1]["scale_labels"][idx1]) and pd.notna(
                reviewer_data[r2]["scale_labels"][idx2]
            ):
                scale_diffs.append(
                    abs(
                        float(reviewer_data[r1]["scale_labels"][idx1])
                        - float(reviewer_data[r2]["scale_labels"][idx2])
                    )
                )

    # Compute scale differences with ground truth
    scale_diffs_gt = []
    for reviewer_num, data in reviewer_data.items():
        for id_, scale_label in zip(data["ids"], data["scale_labels"]):
            if (
                pd.notna(scale_label)
                and id_ in ground_truth_scale_labels
                and id_ in valid_items
            ):
                scale_diffs_gt.append(
                    abs(float(scale_label) - ground_truth_scale_labels[id_])
                )

    # Compute discrete agreement between reviewers
    discrete_agreements = []
    for r1, r2 in reviewer_pairs:
        common_ids = (
            set(reviewer_data[r1]["ids"]) & set(reviewer_data[r2]["ids"]) & valid_items
        )
        for id_ in common_ids:
            idx1 = reviewer_data[r1]["ids"].index(id_)
            idx2 = reviewer_data[r2]["ids"].index(id_)
            if (
                reviewer_data[r1]["discrete_labels"][idx1] is not None
                and reviewer_data[r2]["discrete_labels"][idx2] is not None
            ):
                discrete_agreements.append(
                    reviewer_data[r1]["discrete_labels"][idx1]
                    == reviewer_data[r2]["discrete_labels"][idx2]
                )

    # Compute discrete agreement with ground truth
    discrete_agreements_gt = []
    for reviewer_num, data in reviewer_data.items():
        for id_, discrete_label in zip(data["ids"], data["discrete_labels"]):
            if (
                discrete_label is not None
                and id_ in ground_truth_labels
                and id_ in valid_items
            ):
                discrete_agreements_gt.append(
                    discrete_label == ground_truth_labels[id_]
                )

    # Calculate means and confidence intervals
    def mean_and_ci(
        data: List[Union[float, bool]],
    ) -> Tuple[float, Tuple[float, float]]:
        if not data:
            return 0.0, (0.0, 0.0)
        mean = float(np.mean(data))
        ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=stats.sem(data))
        return mean, (float(ci[0]), float(ci[1]))

    scale_diff_mean, scale_diff_ci = mean_and_ci(scale_diffs)
    scale_diff_gt_mean, scale_diff_gt_ci = mean_and_ci(scale_diffs_gt)
    discrete_agreement_mean, discrete_agreement_ci = mean_and_ci(discrete_agreements)
    discrete_agreement_gt_mean, discrete_agreement_gt_ci = mean_and_ci(
        discrete_agreements_gt
    )

    return (
        scale_diff_mean,
        scale_diff_ci,
        scale_diff_gt_mean,
        scale_diff_gt_ci,
        discrete_agreement_mean,
        discrete_agreement_ci,
        discrete_agreement_gt_mean,
        discrete_agreement_gt_ci,
    )


def analyze_manual_review_results(
    excel_path: Path,
    dataset_name: str,
    use_test: bool = True,
) -> pd.DataFrame:
    """Analyze manual review results from an Excel file.

    Args:
        excel_path: Path to the Excel file containing manual review results
        dataset_name: Name of the dataset being reviewed
        use_test: Whether to use test or dev datasets

    Returns:
        DataFrame with statistics for the dataset
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
        return pd.DataFrame()

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
                    if score_float > 5:
                        discrete_labels.append(Label.HIGH_STAKES)
                    else:
                        discrete_labels.append(Label.LOW_STAKES)
                except (ValueError, TypeError):
                    print(f"Error converting score {score} to float")
                    discrete_labels.append(None)

        reviewer_data[reviewer_num] = {
            "ids": df["ids"].astype(str).tolist(),
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

    # Print detailed per-reviewer pair analysis
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

    # Print agreement with ground truth
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

    # Compute metrics for the summary table
    (
        scale_diff_mean,
        scale_diff_ci,
        scale_diff_gt_mean,
        scale_diff_gt_ci,
        discrete_agreement_mean,
        discrete_agreement_ci,
        discrete_agreement_gt_mean,
        discrete_agreement_gt_ci,
    ) = compute_agreement_metrics(
        reviewer_data, ground_truth_labels, ground_truth_scale_labels, valid_items
    )

    # Create DataFrame with results
    results = pd.DataFrame(
        {
            "Dataset": [dataset_name],
            "Inter-rater Scale Diff": [
                f"{scale_diff_mean:.2f} ({scale_diff_ci[0]:.2f}, {scale_diff_ci[1]:.2f})"
            ],
            "Rater-GT Scale Diff": [
                f"{scale_diff_gt_mean:.2f} ({scale_diff_gt_ci[0]:.2f}, {scale_diff_gt_ci[1]:.2f})"
            ],
            "Inter-rater Agreement": [
                f"{discrete_agreement_mean:.2%} ({discrete_agreement_ci[0]:.2%}, {discrete_agreement_ci[1]:.2%})"
            ],
            "Rater-GT Agreement": [
                f"{discrete_agreement_gt_mean:.2%} ({discrete_agreement_gt_ci[0]:.2%}, {discrete_agreement_gt_ci[1]:.2%})"
            ],
        }
    )

    return results


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

    all_results = []
    for excel_path, dataset_names in datasets_and_files.items():
        for dataset_name in dataset_names:
            print(f"\nAnalyzing {dataset_name} ...".upper())
            results = analyze_manual_review_results(
                excel_path=Path(excel_path),
                dataset_name=dataset_name,
                use_test=True,
            )
            if not results.empty:
                all_results.append(results)

    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        # Add overall row
        # Extract numeric values from the formatted strings
        numeric_metrics = {}
        confidence_intervals = {}

        for col in final_results.columns[1:]:  # Skip the Dataset column
            if "Agreement" in col:
                # For percentage columns, extract the percentage value and confidence interval
                values = final_results[col].str.extract(
                    r"([\d.]+)%\s*\(([\d.]+)%,\s*([\d.]+)%\)"
                )
                # Convert to decimal
                main_values = values[0].astype(float) / 100
                lower_bounds = values[1].astype(float) / 100
                upper_bounds = values[2].astype(float) / 100

                # Cap confidence intervals at 0 and 1
                lower_bounds = np.maximum(lower_bounds, 0)
                upper_bounds = np.minimum(upper_bounds, 1)
            else:
                # For scale difference columns
                values = final_results[col].str.extract(
                    r"([\d.]+)\s*\(([\d.]+),\s*([\d.]+)\)"
                )
                main_values = values[0].astype(float)
                lower_bounds = values[1].astype(float)
                upper_bounds = values[2].astype(float)

            # Compute mean and confidence interval
            mean_value = main_values.mean()
            ci_lower = np.mean(lower_bounds)
            ci_upper = np.mean(upper_bounds)

            numeric_metrics[col] = mean_value
            confidence_intervals[col] = (ci_lower, ci_upper)

        # Create overall row with proper formatting
        overall_row = pd.DataFrame(
            {
                "Dataset": ["Overall"],
                "Inter-rater Scale Diff": [
                    f"{numeric_metrics['Inter-rater Scale Diff']:.2f} ({confidence_intervals['Inter-rater Scale Diff'][0]:.2f}, {confidence_intervals['Inter-rater Scale Diff'][1]:.2f})"
                ],
                "Rater-GT Scale Diff": [
                    f"{numeric_metrics['Rater-GT Scale Diff']:.2f} ({confidence_intervals['Rater-GT Scale Diff'][0]:.2f}, {confidence_intervals['Rater-GT Scale Diff'][1]:.2f})"
                ],
                "Inter-rater Agreement": [
                    f"{numeric_metrics['Inter-rater Agreement']:.2%} ({confidence_intervals['Inter-rater Agreement'][0]:.2%}, {confidence_intervals['Inter-rater Agreement'][1]:.2%})"
                ],
                "Rater-GT Agreement": [
                    f"{numeric_metrics['Rater-GT Agreement']:.2%} ({confidence_intervals['Rater-GT Agreement'][0]:.2%}, {confidence_intervals['Rater-GT Agreement'][1]:.2%})"
                ],
            }
        )
        final_results = pd.concat([final_results, overall_row], ignore_index=True)

        print("\nFinal Results:")
        print(final_results.to_string(index=False))
