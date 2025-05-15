import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models_under_pressure.config import (
    EVALUATE_PROBES_DIR,
    PLOTS_DIR,
    EvalRunConfig,
)
from models_under_pressure.figures.utils import map_dataset_name
from models_under_pressure.interfaces.results import EvaluationResult

# Add this before creating any plots
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    }
)


# Load data from your JSONL file
def load_data(file_path: Path) -> list[dict]:
    with open(EVALUATE_PROBES_DIR / file_path, "r") as file:
        data = [json.loads(line) for line in file]
    return data


# Prepare the data
def prepare_data(
    result: EvaluationResult,
    use_scale_labels: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    dataset_res = result.model_dump()
    dataset_name = result.dataset_name
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
    y_true_list: list[list[int]],
    y_prob_list: list[list[float]],
    scale_labels_list: list[list[float]],
    file_names: list[str],
    config: EvalRunConfig,
    n_bins: int = 10,
    out_path: Path | None = None,
    use_binary_labels: bool = False,
) -> None:
    fig, ax1 = plt.subplots(
        nrows=1,
        figsize=(10, 10),
        # gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=False,
    )
    # fig.subplots_adjust(hspace=0.4)

    # Calibration curve
    # Create bins based on predicted probabilities
    for y_true, y_prob, scale_labels, file_name in zip(
        y_true_list, y_prob_list, scale_labels_list, file_names
    ):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1

        # Calculate mean scale label for each bin
        prob_pred = []
        mean_scale = []
        for i in range(n_bins):
            if use_binary_labels:
                y_axis = y_true
            else:
                y_axis = scale_labels
            mask = bin_indices == i
            if np.any(mask):
                prob_pred.append(np.mean(y_prob[mask]))
                mean_scale.append(np.mean(y_axis[mask]))

        prob_pred = np.array(prob_pred)
        # prob_true = np.array(mean_scale)
        ax1.plot(
            prob_pred,
            mean_scale,
            marker="o",
            linewidth=3,
            label=f"{file_name.title()}",
        )
    ax1.plot([0, 1], [1, 10], linestyle="--", linewidth=3, label="Perfect Calibration")
    ax1.set_xlim(0.0, 1.0)
    if use_binary_labels:
        ax1.set_ylim(0.0, 1.0)
    else:
        ax1.set_ylim(0.0, 10.0)

    # ax1.set_title("Caliberation Curves")
    ax1.set_xlabel("Predicted Probability (Binned)")
    if use_binary_labels:
        ax1.set_ylabel("Mean Binary Label")
    else:
        ax1.set_ylabel("Mean Stakes Rating")
    ax1.grid()
    ax1.legend(title="Probe Calibration")
    # plt.show()
    print(f"Saving {out_path}")
    plt.savefig(out_path)


def plot_stacked_histogram(
    y_true_list: list[list[int]],
    y_prob_list: list[list[float]],
    file_names: list[str],
    config: EvalRunConfig,
    n_bins: int = 10,
) -> None:
    fig, ax2 = plt.subplots(
        nrows=1,
        figsize=(10, 10),
        # gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=False,
    )
    high_stakes_probs = []
    low_stakes_probs = []
    for y_true, y_prob, file_name in zip(y_true_list, y_prob_list, file_names):
        # Separate probabilities for high stakes and low stakes
        high_stakes_probs.extend(
            [prob for prob, label in zip(y_prob, y_true) if label == 1]
        )
        low_stakes_probs.extend(
            [prob for prob, label in zip(y_prob, y_true) if label == 0]
        )
    bins = np.linspace(0, 1, n_bins + 1)
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

    # ax2.set_title(
    #     f"Histogram of Predicted Probabilities for\n{dataset_names[file_name]} dataset"
    # )
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Frequency")
    ax2.grid()
    ax2.legend()

    # save the plots with data name in the same directory
    plot_path = PLOTS_DIR / f"{config.id}_stacked_histogram.pdf"
    print(f"Saving {plot_path}")
    plt.savefig(plot_path)
    plt.close()


def run_calibration(
    evaluate_probe_results_path: Path,
    out_path: Path,
    use_binary_labels: bool = False,
):
    """
    Run calibration analysis with the provided EvalRunConfig.
    If no config is provided, a default one will be created.
    """
    results = [
        EvaluationResult.model_validate_json(line)
        for line in evaluate_probe_results_path.read_text().splitlines()
    ]
    config = results[0].config
    eval_datasets = {
        map_dataset_name(dataset_path): dataset_path
        for dataset_path in config.eval_datasets
        # if map_dataset_name(dataset_path) != "MT"
    }
    y_true_list = []
    y_prob_list = []
    scale_labels_list = []
    for result in results:
        # if map_dataset_name(result.dataset_path) == "MT":
        #     continue
        y_true, y_prob, scale_labels = prepare_data(result, use_scale_labels=True)
        y_true_list.append(y_true)
        y_prob_list.append(y_prob)
        scale_labels_list.append(scale_labels)
    plot_calibration(
        y_true_list,
        y_prob_list,
        scale_labels_list,
        list(eval_datasets.keys()),
        config=config,
        n_bins=10,
        out_path=out_path,
        use_binary_labels=use_binary_labels,
    )
    # plot_stacked_histogram(
    #     y_true_list,
    #     y_prob_list,
    #     list(EVAL_DATASETS_RAW.keys()) + list(TEST_DATASETS_BALANCED.keys()),
    #     config=config,
    #     n_bins=10,
    # )


# Main execution
if __name__ == "__main__":
    run_calibration(
        EVALUATE_PROBES_DIR / "results_for_calibration.jsonl",
        out_path=PLOTS_DIR / "calibration_all_binary_labels.pdf",
        use_binary_labels=True,
    )
    run_calibration(
        EVALUATE_PROBES_DIR / "results_attention_test_1.jsonl",
        out_path=PLOTS_DIR / "attention_test_1_calibration.pdf",
        use_binary_labels=False,
    )

    run_calibration(
        EVALUATE_PROBES_DIR / "results_softmax_test_1.jsonl",
        out_path=PLOTS_DIR / "softmax_test_1_calibration.pdf",
        use_binary_labels=False,
    )

    run_calibration(
        EVALUATE_PROBES_DIR / "results_mean_test_1.jsonl",
        out_path=PLOTS_DIR / "mean_test_1_calibration.pdf",
        use_binary_labels=False,
    )
