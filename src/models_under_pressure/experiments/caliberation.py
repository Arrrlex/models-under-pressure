import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from models_under_pressure.config import (
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    MANUAL_UPSAMPLED_DATASET_PATH,
    PLOTS_DIR,
    EvalRunConfig,
)


def load_results(file_path: Path) -> list[dict]:
    """Load results from a JSON file containing multiple datasets."""
    with open(file_path, "r") as f:
        results = [json.loads(line) for line in f]
    return results


def prepare_data(
    result: dict, use_scale_labels: bool = False
) -> tuple[list[int], list[float]]:
    """Extract ground truth labels and output scores from a single result entry."""
    if use_scale_labels:
        y_true = result["ground_truth_scale_labels"]
    else:
        if "output_labels" in result and result["output_labels"] is not None:
            y_true = result["output_labels"]
        else:
            y_true = [0 if score < 0.5 else 1 for score in result["output_scores"]]
    y_prob = result["output_scores"]
    return y_true, y_prob


def plot_calibration(
    y_true: list[int],
    y_prob: list[float],
    dataset_name: str,
    model_name: str,
    layer: int,
    n_bins: int = 10,
) -> None:
    """Plot calibration curve and histogram for a single dataset."""
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, figsize=(8, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax1.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model Calibration")
    ax1.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    ax1.set_title(f"Calibration Curve - {dataset_name}")
    ax1.set_xlabel("Predicted Probability (Binned)")
    ax1.set_ylabel("Mean Observed Label")
    ax1.grid()
    ax1.legend()

    # Histogram
    ax2.hist(y_prob, range=(0, 1), bins=n_bins, edgecolor="black")
    ax2.set_title("Histogram of Predicted Probabilities")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Frequency")
    ax2.grid()

    plt.suptitle(f"{model_name} - Layer {layer}")
    plt.tight_layout()

    # Save plot
    output_path = (
        PLOTS_DIR
        / f"calibration_{dataset_name}_{Path(eval_run_config.output_filename).stem}.png"
    )
    plt.savefig(output_path)
    plt.close()


def run_calibration(results_file: Path) -> None:
    """Generate calibration plots for each dataset in the results file."""
    results = load_results(results_file)

    for result in results:
        dataset_name = result["dataset_name"]
        model_name = result["model_name"].split("/")[-1]
        layer = result["method_details"]["layer"]

        y_true, y_prob = prepare_data(result, use_scale_labels=False)
        plot_calibration(
            y_true=y_true,
            y_prob=y_prob,
            dataset_name=dataset_name,
            model_name=model_name,
            layer=layer,
            n_bins=10,
        )


if __name__ == "__main__":
    eval_run_config = EvalRunConfig(
        dataset_path=MANUAL_UPSAMPLED_DATASET_PATH,
        layer=22,
        model_name=LOCAL_MODELS["llama-70b"],
    )

    results_file = EVALUATE_PROBES_DIR / eval_run_config.output_filename
    run_calibration(results_file)
