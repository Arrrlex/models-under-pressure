import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models_under_pressure.config import (
    EVALUATE_PROBES_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    EvalRunConfig,
)
from models_under_pressure.figures.utils import map_dataset_name
from models_under_pressure.interfaces.results import EvaluationResult

# Add this before creating any plots
plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
    }
)


# Load data from your JSONL file
def load_data(file_path: Path) -> list[dict]:
    with open(EVALUATE_PROBES_DIR / file_path, "r") as file:
        data = [json.loads(line) for line in file]
    return data


# Prepare the data
def prepare_data(
    result: EvaluationResult | dict,
    use_scale_labels: bool = False,
) -> tuple[list[int], list[float], list[float], float]:
    # Handle both EvaluationResult and raw dict inputs
    if isinstance(result, EvaluationResult):
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

        if dataset_name == "manual":
            scale_labels = np.array(dataset_res["ground_truth_labels"])  # type: ignore
        else:
            scale_labels = np.array(dataset_res["ground_truth_scale_labels"])  # type: ignore
    else:
        # Handle finetuned baseline results or prompting baseline results
        dataset_res = result
        if "high_stakes_scores" in dataset_res and "low_stakes_scores" in dataset_res:
            # Handle prompting baseline results
            y_prob = dataset_res["high_stakes_scores"]  # type: ignore
            y_true = dataset_res["labels"]  # type: ignore
            scale_labels = np.array(dataset_res["ground_truth_scale_labels"])  # type: ignore
        else:
            # Handle finetuned baseline results
            y_prob = dataset_res["scores"]  # type: ignore
            y_true = dataset_res["labels"]  # type: ignore
            scale_labels = np.array(dataset_res["ground_truth_scale_labels"])  # type: ignore

    # Convert y_prob and y_true to numpy arrays for easier manipulation
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    # Calculate Brier score
    if not use_scale_labels:
        binary_true = np.array([1 if label > 5 else 0 for label in scale_labels])
    else:
        binary_true = y_true
    brier_score = np.mean((y_prob - binary_true) ** 2)

    # Sort probabilities and corresponding scale labels
    sort_indices = np.argsort(y_prob)
    y_prob = y_prob[sort_indices]
    scale_labels = scale_labels[sort_indices]
    y_true = y_true[sort_indices]
    return y_true, y_prob, scale_labels, brier_score  # type: ignore


# Plot calibration curve and histogram
def plot_calibration(
    y_true_list: list[list[int]],
    y_prob_list: list[list[float]],
    scale_labels_list: list[list[float]],
    file_names: list[str],
    out_path: Path,
    n_bins: int = 10,
    use_binary_labels: bool = False,
    with_legend: bool = True,
) -> tuple[Path, Path]:
    fig, ax1 = plt.subplots(
        nrows=1,
        figsize=(10, 10),
        # gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=False,
    )
    # fig.subplots_adjust(hspace=0.4)

    # Calculate mean Brier score across all datasets
    brier_scores = []
    for y_true, y_prob in zip(y_true_list, y_prob_list):
        # Convert to binary labels if needed
        if not use_binary_labels:
            y_true = np.array([1 if label > 5 else 0 for label in y_true])
        else:
            y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        brier_score = np.mean((y_prob - y_true) ** 2)
        brier_scores.append(brier_score)
    mean_brier_score = np.mean(brier_scores)

    # Calibration curve
    # Create bins based on predicted probabilities
    plot_data = {
        "datasets": {},
        "mean_brier_score": float(mean_brier_score),
    }
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
            linewidth=4,
            label=f"{file_name.title()}",
        )

        # Store data for JSON output
        plot_data["datasets"][file_name] = {
            "predicted_probabilities": prob_pred.tolist(),
            "mean_scale_labels": mean_scale,
            "brier_score": float(brier_scores[len(plot_data["datasets"])]),
        }

    ax1.plot([0, 1], [1, 10], linestyle="--", linewidth=4, label="Perfect Calibration")
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
    if with_legend:
        ax1.legend(title="Probe Calibration")

    # Save plot
    print(f"Saving {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)

    # Save data as JSON
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    print(f"Saving {json_path}")

    return out_path, json_path


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
        bins=bins.tolist(),
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


def plot_brier_scores(
    classifiers: dict[str, Path],
    out_path: Path,
) -> None:
    """Plot mean Brier scores for each classifier."""
    plt.figure(figsize=(20, 12))

    mean_brier_scores = []
    classifier_names = []

    for name, path in classifiers.items():
        # Load the calibration results
        with open(path) as f:
            data = json.load(f)

        mean_brier_score = data["mean_brier_score"]
        mean_brier_scores.append(mean_brier_score)
        classifier_names.append(name)

    # Create bar plot
    plt.bar(classifier_names, mean_brier_scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Brier Score")
    plt.title("Mean Brier Scores by Classifier")
    plt.grid(axis="y")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    print(f"Saving {out_path}")
    plt.savefig(out_path)
    plt.close()


def run_calibration(
    evaluate_probe_results_path: Path,
    out_path: Path,
    use_binary_labels: bool = False,
    with_legend: bool = True,
):
    """
    Run calibration analysis with the provided EvalRunConfig.
    If no config is provided, a default one will be created.
    """
    # Try to load as EvaluationResult first, fall back to raw JSON if that fails
    results = []
    for line in evaluate_probe_results_path.read_text().splitlines():
        try:
            results.append(EvaluationResult.model_validate_json(line))
        except Exception:
            results.append(json.loads(line))

    # Get config from first result if it's an EvaluationResult
    config = results[0].config if isinstance(results[0], EvaluationResult) else None

    # For finetuned baseline results, use the dataset name from the file path
    eval_datasets = {}
    if config:
        eval_datasets = {
            map_dataset_name(dataset_path): dataset_path
            for dataset_path in config.eval_datasets
        }
    else:
        # For baseline results, use dataset_name from each result
        eval_datasets = {
            result["dataset_name"]: result["dataset_path"] for result in results
        }

    y_true_list = []
    y_prob_list = []
    scale_labels_list = []
    for result in results:
        y_true, y_prob, scale_labels, _ = prepare_data(result, use_scale_labels=True)
        y_true_list.append(y_true)
        y_prob_list.append(y_prob)
        scale_labels_list.append(scale_labels)

    out_path, json_path = plot_calibration(
        y_true_list,
        y_prob_list,
        scale_labels_list,
        list(eval_datasets.keys()),
        n_bins=10,
        out_path=out_path,
        use_binary_labels=use_binary_labels,
        with_legend=with_legend,
    )

    return out_path, json_path


# Main execution
if __name__ == "__main__":
    WITH_LEGEND = False
    calibration_plots_dir = PLOTS_DIR / "calibration_plots"
    calibration_plots_dir.mkdir(parents=True, exist_ok=True)

    probes_dir = EVALUATE_PROBES_DIR
    finetuned_dir = RESULTS_DIR / "finetuning_baselines_test_performance"
    finetuned_v2_dir = RESULTS_DIR / "finetuning_baselines_test_performance_v2"
    prompting_dir = RESULTS_DIR / "prompting_baselines_test_performance"

    results_paths = [
        *probes_dir.glob("*test_1.jsonl"),
        finetuned_v2_dir / "finetuning_gemma_1b_test_optimized_2.jsonl",
        finetuned_v2_dir / "finetuning_gemma_12b_test_optimized_2.jsonl",
        finetuned_v2_dir / "finetuning_llama_1b_test_optimized_2.jsonl",
        finetuned_v2_dir / "finetuning_llama_8b_test_optimized_2.jsonl",
        prompting_dir / "baseline_gemma-1b_prompt_at_end.jsonl",
        prompting_dir / "baseline_gemma-12b.jsonl",
        prompting_dir / "baseline_gemma-27b.jsonl",
        prompting_dir / "baseline_llama-1b.jsonl",
        prompting_dir / "baseline_llama-8b_default.jsonl",
        prompting_dir / "baseline_llama-70b.jsonl",
    ]

    calibration_jsons = {}
    for results_path in results_paths:
        if "prompting" in str(results_path):
            results_type = "prompted"
        elif "finetuning" in str(results_path):
            results_type = "finetuned"
        elif "probe" in str(results_path):
            results_type = "probe"

        if results_type in ["finetuned", "prompted"]:
            match = re.search(
                r"(llama|gemma)[-_](\d{1,2})b", results_path.stem, re.IGNORECASE
            )
            if match:
                model_name = match.group().lower()
            else:
                model_name = results_path.stem
            results_id = f"{results_type}_{model_name}"
        elif results_type == "probe":
            match = re.search(r"results_(\w+)_test_1.jsonl", str(results_path))
            if match:
                probe_type = match.group(1)
                results_id = f"probe_{probe_type}"
            else:
                results_id = results_path.stem
            calibration_jsons[results_id] = results_path

        if WITH_LEGEND:
            legend_suffix = "with_legend"
        else:
            legend_suffix = "without_legend"

        plot_path, json_path = run_calibration(
            results_path,
            out_path=calibration_plots_dir
            / f"calibration_{results_id}_{legend_suffix}.png",
            use_binary_labels=False,
            with_legend=WITH_LEGEND,
        )

        calibration_jsons[results_id] = json_path

    plot_brier_scores(
        calibration_jsons,
        out_path=calibration_plots_dir / "brier_scores.png",
    )
