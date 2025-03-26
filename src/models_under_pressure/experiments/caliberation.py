import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models_under_pressure.config import (
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    PLOTS_DIR,
    EvalRunConfig,
)
from models_under_pressure.interfaces.probes import ProbeSpec

# Add this before creating any plots
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)


# Load data from your JSONL file
def load_data(file_path: Path) -> list[dict]:
    with open(EVALUATE_PROBES_DIR / file_path, "r") as file:
        data = [json.loads(line) for line in file]
    return data


# Prepare the data
def prepare_data(
    data: list[dict],
    dataset_name: str,
    config: EvalRunConfig,
    use_scale_labels: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    dataset_res = [
        entry
        if (
            entry["config"]["id"] == config.id
            and entry["dataset_name"] == dataset_name
            and entry["metrics"]["layer"] == config.layer
            and entry["config"]["model_name"] == config.model_name
            and entry["config"]["max_samples"] == config.max_samples
        )
        else None
        for entry in data
    ]
    # extract the not none entry first

    dataset_res = [entry for entry in dataset_res if entry is not None][-1]
    y_prob = dataset_res["output_scores"]  # type: ignore
    if use_scale_labels:
        if dataset_name == "manual":
            print(
                "Cannot use scale labels for manual dataset, using output labels instead"
            )
            y_true = dataset_res["output_labels"]  # type: ignore
        else:
            y_true = [
                1 if entry > 5 else 0
                for entry in dataset_res["ground_truth_scale_labels"]
            ]  # type: ignore
    else:
        y_true = dataset_res["ground_truth_labels"]  # type: ignore

    # Convert y_prob and y_true to numpy arrays for easier manipulation
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    if dataset_name == "manual":
        scale_labels = np.array(dataset_res["ground_truth_labels"])  # type: ignore
    else:
        scale_labels = np.array(dataset_res["ground_truth_scale_labels"])  # type: ignore

    # Sort probabilities and corresponding scale labels
    sort_indices = np.argsort(y_prob)
    y_prob = y_prob[sort_indices]
    scale_labels = scale_labels[sort_indices]
    y_true = y_true[sort_indices]
    return y_true, y_prob, scale_labels  # type: ignore


# Plot calibration curve and histogram
def plot_calibration(
    y_true: list[int],
    y_prob: list[float],
    scale_labels: list[float],
    file_name: str,
    config: EvalRunConfig,
    n_bins: int = 10,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        figsize=(9, 10),
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=False,
    )
    fig.subplots_adjust(hspace=0.4)

    dataset_names = {
        "manual": "Manual",
        "anthropic": "Anthropic HH-RLHF",
        "mt": "Medical Transcriptions",
        "mts": "Medical Transcriptions (Clinical Dialogues)",
        "toolace": "ToolACE",
    }

    # Calibration curve
    # Create bins based on predicted probabilities

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1

    # Calculate mean scale label for each bin
    prob_pred = []
    mean_scale = []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            prob_pred.append(np.mean(y_prob[mask]))
            mean_scale.append(np.mean(scale_labels[mask]))

    prob_pred = np.array(prob_pred)
    # prob_true = np.array(mean_scale)
    ax1.plot(prob_pred, mean_scale, marker="o", linewidth=2, label="Probe Caliberation")
    ax1.plot([0, 1], [1, 10], linestyle="--", label="Perfect Caliberation")
    ax1.set_title(f"Caliberation Curve for {dataset_names[file_name]} dataset")
    ax1.set_xlabel("Predicted Probability (Binned)")
    ax1.set_ylabel("Mean Observed Label")
    ax1.grid()
    ax1.legend()

    # Stacked histogram showing high stakes (1) and low stakes (0) with different colors
    bins = np.linspace(0, 1, n_bins + 1)

    # Separate probabilities for high stakes and low stakes
    high_stakes_probs = [prob for prob, label in zip(y_prob, y_true) if label == 1]
    low_stakes_probs = [prob for prob, label in zip(y_prob, y_true) if label == 0]

    # Plot stacked histogram
    ax2.hist(
        [low_stakes_probs, high_stakes_probs],
        bins=bins,
        stacked=True,
        color=["green", "red"],
        label=["Low Stakes", "High Stakes"],
        edgecolor="black",
        alpha=0.7,
    )

    ax2.set_title(
        f"Histogram of Predicted Probabilities for\n{dataset_names[file_name]} dataset"
    )
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Frequency")
    ax2.grid()

    # save the plots with data name in the same directory
    plt.savefig(PLOTS_DIR / f"{config.id}_{file_name}_calibration.png")
    plt.close()


def run_calibration(config: EvalRunConfig):
    """
    Run calibration analysis with the provided EvalRunConfig.
    If no config is provided, a default one will be created.
    """
    for eval_dataset in EVAL_DATASETS.keys():
        data = load_data(EVALUATE_PROBES_DIR / "raw_results" / config.output_filename)
        y_true, y_prob, scale_labels = prepare_data(
            data, eval_dataset, config=config, use_scale_labels=True
        )
        plot_calibration(
            y_true, y_prob, scale_labels, eval_dataset, config=config, n_bins=10
        )


# Main execution
if __name__ == "__main__":
    id_used_in_eval = "raw_caliberation_all"
    model_name = LOCAL_MODELS["llama-70b"]
    layer = 31
    run_calibration(
        EvalRunConfig(
            id=id_used_in_eval,
            model_name=model_name,
            layer=layer,
            max_samples=None,
            probe_spec=ProbeSpec(
                name="pytorch_per_token_probe",
            ),
        ),
    )
