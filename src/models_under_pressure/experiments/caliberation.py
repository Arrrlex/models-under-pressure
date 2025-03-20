import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from models_under_pressure.config import (
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    PLOTS_DIR,
)


# Load data from your JSONL file
def load_data(file_path: Path) -> list[dict]:
    with open(EVALUATE_PROBES_DIR / file_path, "r") as file:
        data = [json.loads(line) for line in file]
    return data


# Prepare the data
def prepare_data(data: list[dict], dataset_name: str) -> tuple[list[int], list[float]]:
    dataset_res = [
        data if entry["dataset_name"] == dataset_name else None for entry in data
    ]
    # extract the not none entry first
    dataset_res = [entry for entry in dataset_res if entry is not None][0][0]
    y_prob = dataset_res["output_scores"]
    y_true = [1 if entry["scale_labels"] > 5 else 0 for entry in dataset_res[""]]
    return y_true, y_prob


# Plot calibration curve and histogram
def plot_calibration(
    y_true: list[int], y_prob: list[float], file_name: str, n_bins: int = 10
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, figsize=(8, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax1.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model Calibration")
    ax1.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    ax1.set_title(f"Calibration Curve for {file_name}")
    ax1.set_xlabel("Predicted Probability (Binned)")
    ax1.set_ylabel("Mean Observed Label")
    ax1.grid()
    ax1.legend()

    # Histogram
    ax2.hist(y_prob, range=(0, 1), bins=n_bins, edgecolor="black")
    ax2.set_title(f"Histogram of Predicted Probabilities for {file_name}")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Frequency")
    ax2.grid()

    # save the plots with data name in the same directory
    plt.savefig(PLOTS_DIR / f"{type}_{file_name}_calibration.png")
    plt.close()


def run_calibration(type: str, model_name: str, layer: int):
    """
    Run calibration analysis with the provided EvalRunConfig.
    If no config is provided, a default one will be created.
    """
    for eval_dataset in EVAL_DATASETS.keys():
        data = load_data(EVALUATE_PROBES_DIR / type)
        y_true, y_prob = prepare_data(data, eval_dataset)
        plot_calibration(y_true, y_prob, eval_dataset, n_bins=10)


# Main execution
if __name__ == "__main__":
    type = "upsampled_train_Llama-3_layer22_fig2.json"
    model_name = LOCAL_MODELS["llama-70b"]
    layer = 22
    run_calibration(type, model_name, layer)
