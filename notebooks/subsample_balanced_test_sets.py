from pathlib import Path
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
    subsample_balanced_subset,
)
from models_under_pressure.config import TEST_EVALS_DIR


def subsample_balanced_dataset(in_path: str | Path, out_path: str | Path) -> None:
    """
    Load a dataset and subsample it to have equal numbers of each label type (high-stakes, low-stakes, ambiguous),
    with a maximum of 2000 rows total.

    Args:
        in_path: Path to the input dataset file
        out_path: Path where the subsampled dataset will be saved
    """
    # Convert paths to Path objects
    in_path = Path(in_path)
    out_path = Path(out_path)

    # Load the dataset
    dataset = LabelledDataset.load_from(in_path)

    # Calculate the maximum number of samples per class to stay under 2000 total

    # Subsample the dataset to have equal numbers of each label
    balanced_dataset = subsample_balanced_subset(dataset, include_ambiguous=True)

    n_per_class = min(len(balanced_dataset) // 3, 400)
    balanced_dataset = subsample_balanced_subset(
        balanced_dataset, n_per_class=n_per_class, include_ambiguous=True
    )

    # Save the balanced dataset
    if out_path.exists():
        print(f"Dataset already exists at {out_path}, skipping")
        return
    else:
        balanced_dataset.save_to(out_path)

    # Print the label distribution
    balanced_dataset.print_label_distribution()


if __name__ == "__main__":
    subsample_balanced_dataset(
        in_path=TEST_EVALS_DIR / "anthropic_test_raw_apr_23.jsonl",
        out_path=TEST_EVALS_DIR / "anthropic_test_balanced_ambiguous_may_7.jsonl",
    )

    subsample_balanced_dataset(
        in_path=TEST_EVALS_DIR / "redteaming_test_raw_apr_22.jsonl",
        out_path=TEST_EVALS_DIR / "redteaming_test_balanced_ambiguous_may_7.jsonl",
    )

    # subsample_balanced_dataset(
    #     in_path=TEST_EVALS_DIR / "mental_health_test_raw_apr_22.jsonl",
    #     out_path=TEST_EVALS_DIR / "mental_health_test_balanced_ambiguous_apr_22.jsonl",
    # )

    # subsample_balanced_dataset(
    #     in_path=TEST_EVALS_DIR / "mt_test_raw_apr_30.jsonl",
    #     out_path=TEST_EVALS_DIR / "mt_test_balanced_ambiguous_apr_30.jsonl",
    # )

    # subsample_balanced_dataset(
    #     in_path=TEST_EVALS_DIR / "mts_test_raw_apr_22.jsonl",
    #     out_path=TEST_EVALS_DIR / "mts_test_balanced_ambiguous_apr_22.jsonl",
    # )

    # subsample_balanced_dataset(
    #     in_path=TEST_EVALS_DIR / "toolace_test_raw_apr_22.jsonl",
    #     out_path=TEST_EVALS_DIR / "toolace_test_balanced_ambiguous_apr_22.jsonl",
    # )
