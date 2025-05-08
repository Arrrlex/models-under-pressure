from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from models_under_pressure.config import RESULTS_DIR
from models_under_pressure.interfaces.results import (
    EvaluationResult,
    FinetunedBaselineResults,
    LikelihoodBaselineResults,
)


def load_results(
    probe_results_path: Path,
    dataset_name: str,
    finetuned_results_path: Optional[Path] = None,
    continuation_results_path: Optional[Path] = None,
) -> tuple[
    EvaluationResult,
    Optional[FinetunedBaselineResults],
    Optional[LikelihoodBaselineResults],
]:
    """Load results from JSONL files for a specific dataset.

    Args:
        probe_results_path: Path to probe results file
        dataset_name: Name of the dataset to filter results for
        finetuned_results_path: Optional path to finetuned baseline results file
        continuation_results_path: Optional path to continuation baseline results file

    Returns:
        Tuple of loaded results for the specified dataset
    """
    probe_results = None
    with open(probe_results_path) as f:
        for line in f:
            result = EvaluationResult.model_validate_json(line)
            if result.dataset_name == dataset_name:
                probe_results = result
                break

    if probe_results is None:
        raise ValueError(
            f"No results found for dataset '{dataset_name}' in probe results"
        )

    finetuned_results = None
    if finetuned_results_path:
        with open(finetuned_results_path) as f:
            for line in f:
                result = FinetunedBaselineResults.model_validate_json(line)
                if result.dataset_name == dataset_name:
                    finetuned_results = result
                    break

    continuation_results = None
    if continuation_results_path:
        with open(continuation_results_path) as f:
            for line in f:
                result = LikelihoodBaselineResults.model_validate_json(line)
                if result.dataset_name == dataset_name:
                    continuation_results = result
                    break

    return probe_results, finetuned_results, continuation_results


def compute_roc_curve(
    y_true: List[int], y_score: List[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve and AUC score.

    Args:
        y_true: Ground truth labels
        y_score: Prediction scores

    Returns:
        Tuple of (fpr, tpr, auc_score)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, roc_auc


def plot_roc_curves(
    probe_results: EvaluationResult,
    dataset_name: str,
    finetuned_results: Optional[FinetunedBaselineResults] = None,
    continuation_results: Optional[LikelihoodBaselineResults] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Plot ROC curves comparing different methods.

    Args:
        probe_results: Results from probe evaluation
        dataset_name: Name of the dataset being plotted
        finetuned_results: Optional results from finetuned baseline
        continuation_results: Optional results from continuation baseline
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Plot probe results
    if (
        probe_results.output_scores is not None
        and probe_results.ground_truth_labels is not None
    ):
        fpr, tpr, roc_auc = compute_roc_curve(
            probe_results.ground_truth_labels, probe_results.output_scores
        )
        plt.plot(
            fpr,
            tpr,
            label=f"Probe (AUC = {roc_auc:.3f})",
            color="blue",
        )

    # Plot finetuned baseline results
    if finetuned_results is not None:
        fpr, tpr, roc_auc = compute_roc_curve(
            finetuned_results.ground_truth, finetuned_results.scores
        )
        plt.plot(
            fpr,
            tpr,
            label=f"Finetuned Baseline (AUC = {roc_auc:.3f})",
            color="green",
        )

    # Plot continuation baseline results
    if continuation_results is not None:
        # Use high stakes scores for ROC curve
        fpr, tpr, roc_auc = compute_roc_curve(
            continuation_results.ground_truth, continuation_results.high_stakes_scores
        )
        plt.plot(
            fpr,
            tpr,
            label=f"Continuation Baseline (AUC = {roc_auc:.3f})",
            color="red",
        )

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", label="Random")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves Comparison - {dataset_name}")
    plt.legend(loc="lower right")
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot ROC curves comparing different methods"
    )
    parser.add_argument(
        "--probe-results", type=Path, required=True, help="Path to probe results file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to plot results for",
    )
    parser.add_argument(
        "--finetuned-results", type=Path, help="Path to finetuned baseline results file"
    )
    parser.add_argument(
        "--continuation-results",
        type=Path,
        help="Path to continuation baseline results file",
    )
    parser.add_argument("--output", type=Path, help="Path to save the plot")

    args = parser.parse_args()

    probe_results, finetuned_results, continuation_results = load_results(
        args.probe_results,
        args.dataset,
        args.finetuned_results,
        args.continuation_results,
    )

    plot_roc_curves(
        probe_results,
        args.dataset,
        finetuned_results,
        continuation_results,
        args.output,
    )


if __name__ == "__main__":
    probe_results_path = (
        Path(RESULTS_DIR) / "monitoring_cascade_neurips" / "probe_results.jsonl"
    )
    finetuned_results_path = (
        Path(RESULTS_DIR)
        / "monitoring_cascade_neurips"
        / "finetuning_gemma12b_test.jsonl"
    )
    continuation_results_path = (
        Path(RESULTS_DIR) / "monitoring_cascade_neurips" / "baseline_gemma-12b.jsonl"
    )
    dataset_name = "monitoring_cascade_neurips"  # Example dataset name
    probe_results, finetuned_results, continuation_results = load_results(
        probe_results_path,
        dataset_name,
        finetuned_results_path,
        continuation_results_path,
    )

    plot_roc_curves(
        probe_results,
        dataset_name,
        finetuned_results,
        continuation_results,
        output_path=Path(RESULTS_DIR) / "monitoring_cascade_neurips" / "roc_curves.png",
    )
