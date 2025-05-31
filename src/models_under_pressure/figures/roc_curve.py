from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from pydantic import BaseModel, ValidationError
from sklearn.metrics import roc_curve

from models_under_pressure.config import RESULTS_DIR
from models_under_pressure.figures.utils import map_dataset_name
from models_under_pressure.interfaces.results import (
    EvaluationResult,
    FinetunedBaselineDataEfficiencyResults,
    FinetunedBaselineResults,
    LikelihoodBaselineResults,
)


class ROCCurveFile(BaseModel):
    """Information about a results file for ROC curve plotting."""

    name: str
    type: Literal["probe", "finetuned_baseline", "prompted_baseline"]
    path: Path


# Color schemes matching compare_probes_plot.py and probes_vs_baseline_plot.py
PROBE_COLORS = {
    "Attention Probe": "#FF7F0E",  # Orange
    "Softmax Probe": "#1F77B4",  # Blue
    "Last Token Probe": "#2CA02C",  # Green
    "Max Probe": "#9467BD",  # Purple
    "Mean Probe": "#8C564B",  # Brown
    "Rolling Mean Max Probe": "#E377C2",  # Pink
}

# Colors differentiated by model family within method types
METHOD_COLORS = {
    # Finetuned: matching probes_vs_baseline_plot.py exactly
    "finetuned_llama": "#008000",  # Dark green
    "finetuned_gemma": "#32CD32",  # Lime green
    # Prompted: matching probes_vs_baseline_plot.py exactly (continuation colors)
    "prompted_llama": "#008080",  # Teal
    "prompted_gemma": "#00CED1",  # Dark turquoise
}

# Line styles by parameter count - more solid for larger models
MODEL_LINE_STYLES = {
    "1b": ":",  # Dotted (smallest - 1B)
    "8b": "-.",  # Dash-dot (8B)
    "12b": "--",  # Dashed (12B)
    "27b": "-",  # Solid (27B)
    "70b": "-",  # Solid (70B - largest)
}


def get_method_style(name: str) -> tuple[str, str]:
    """Get the appropriate color and line style for a method based on its name.

    Returns:
        Tuple of (color, linestyle)
    """
    # Check if it's a probe - probes use solid lines
    if name in PROBE_COLORS:
        return PROBE_COLORS[name], "-"

    # Check for finetuned baselines
    if "finetuned" in name.lower():
        # Determine color based on model family
        if "llama" in name.lower():
            color = METHOD_COLORS["finetuned_llama"]
        else:  # gemma
            color = METHOD_COLORS["finetuned_gemma"]

        # All finetuned use solid lines now
        return color, "-"

    # Check for prompted baselines
    if "prompted" in name.lower():
        # Determine color based on model family
        if "llama" in name.lower():
            color = METHOD_COLORS["prompted_llama"]
        else:  # gemma
            color = METHOD_COLORS["prompted_gemma"]

        # All prompted use solid lines now
        return color, "-"

    # Default fallback
    return "#666666", "-"


def get_method_color(name: str) -> str:
    """Get the appropriate color for a method based on its name."""
    color, _ = get_method_style(name)
    return color


def get_roc_curve_data_for_probe(
    probe_results: list[EvaluationResult],
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """
    Get the ROC curve data from the probe results.

    Returns:
        Dictionary mapping dataset names to (false_positive_rates, true_positive_rates, thresholds)
    """
    # Group results by dataset name
    dataset_results = {}

    for result in probe_results:
        dataset_name = map_dataset_name(result.dataset_name)
        if result.output_scores is not None and result.ground_truth_labels is not None:
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = {"scores": [], "labels": []}

            dataset_results[dataset_name]["scores"].extend(result.output_scores)
            dataset_results[dataset_name]["labels"].extend(result.ground_truth_labels)

    # Compute ROC curves for each dataset
    roc_data = {}
    for dataset_name, data in dataset_results.items():
        if data["scores"] and data["labels"]:
            fpr, tpr, thresholds = roc_curve(data["labels"], data["scores"])
            roc_data[dataset_name] = (fpr.tolist(), tpr.tolist(), thresholds.tolist())
        else:
            roc_data[dataset_name] = ([], [], [])

    return roc_data


def get_roc_curve_data_for_finetuned_baseline(
    baseline_results: list[
        FinetunedBaselineDataEfficiencyResults | FinetunedBaselineResults
    ],
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """
    Get the ROC curve data from the finetuned baseline results.

    Returns:
        Dictionary mapping dataset names to (false_positive_rates, true_positive_rates, thresholds)
    """
    # Group results by dataset name
    dataset_results = {}

    for result in baseline_results:
        dataset_name = map_dataset_name(result.dataset_name)
        if result.scores is not None and result.ground_truth is not None:
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = {"scores": [], "labels": []}

            dataset_results[dataset_name]["scores"].extend(result.scores)
            dataset_results[dataset_name]["labels"].extend(result.ground_truth)

    # Compute ROC curves for each dataset
    roc_data = {}
    for dataset_name, data in dataset_results.items():
        if data["scores"] and data["labels"]:
            fpr, tpr, thresholds = roc_curve(data["labels"], data["scores"])
            roc_data[dataset_name] = (fpr.tolist(), tpr.tolist(), thresholds.tolist())
        else:
            roc_data[dataset_name] = ([], [], [])

    return roc_data


def get_roc_curve_data_for_prompted_baseline(
    baseline_results: list[LikelihoodBaselineResults],
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """
    Get the ROC curve data from the prompted baseline results.
    We're looking at the high stakes score as mentioned in the prompt.

    Returns:
        Dictionary mapping dataset names to (false_positive_rates, true_positive_rates, thresholds)
    """
    # Group results by dataset name
    dataset_results = {}

    for result in baseline_results:
        dataset_name = map_dataset_name(result.dataset_name)
        if result.high_stakes_scores is not None and result.ground_truth is not None:
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = {"scores": [], "labels": []}

            dataset_results[dataset_name]["scores"].extend(result.high_stakes_scores)
            dataset_results[dataset_name]["labels"].extend(result.ground_truth)

    # Compute ROC curves for each dataset
    roc_data = {}
    for dataset_name, data in dataset_results.items():
        if data["scores"] and data["labels"]:
            fpr, tpr, thresholds = roc_curve(data["labels"], data["scores"])
            roc_data[dataset_name] = (fpr.tolist(), tpr.tolist(), thresholds.tolist())
        else:
            roc_data[dataset_name] = ([], [], [])

    return roc_data


def get_roc_curve_data(
    type: Literal["probe", "finetuned_baseline", "prompted_baseline"],
    path: Path,
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """
    Get the ROC curve data from the given path.

    Args:
        type: The type of results in the JSONL file
        path: Path to the JSONL file containing results

    Returns:
        Dictionary mapping dataset names to (false_positive_rates, true_positive_rates, thresholds)
    """
    # Read all lines from the file
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return {}

    # Parse lines based on the specified type
    if type == "probe":
        probe_results = []
        for line in lines:
            try:
                result = EvaluationResult.model_validate_json(line)
                probe_results.append(result)
            except ValidationError:
                continue
        return get_roc_curve_data_for_probe(probe_results)

    elif type == "finetuned_baseline":
        finetuned_baseline_results = []
        for line in lines:
            try:
                # Try FinetunedBaselineDataEfficiencyResults first
                result = FinetunedBaselineDataEfficiencyResults.model_validate_json(
                    line
                )
                finetuned_baseline_results.append(result)
            except ValidationError:
                try:
                    # Fall back to FinetunedBaselineResults
                    result = FinetunedBaselineResults.model_validate_json(line)
                    finetuned_baseline_results.append(result)
                except ValidationError:
                    continue
        return get_roc_curve_data_for_finetuned_baseline(finetuned_baseline_results)

    elif type == "prompted_baseline":
        likelihood_baseline_results = []
        for line in lines:
            try:
                result = LikelihoodBaselineResults.model_validate_json(line)
                likelihood_baseline_results.append(result)
            except ValidationError:
                continue
        return get_roc_curve_data_for_prompted_baseline(likelihood_baseline_results)


def plot_roc_curves(
    results: list[ROCCurveFile],
    dataset_name: str,
    output_path: Path,
    show_legend: bool = True,
) -> None:
    """
    Plot the ROC curves for the given results on a specific dataset.

    Args:
        results: List of ROC curve file information
        dataset_name: Name of the dataset to plot ROC curves for
        output_path: Path to save the plot
        show_legend: Whether to show the legend (default: True)
    """
    plt.figure(figsize=(6, 6))  # Square figure

    # Add random classifier diagonal line
    plt.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.5, linewidth=2.5)

    # Plot each ROC curve
    for result_file in results:
        dataset_roc_data = get_roc_curve_data(result_file.type, result_file.path)

        # Get ROC data for the specific dataset
        if dataset_name in dataset_roc_data:
            fpr, tpr, thresholds = dataset_roc_data[dataset_name]
            if fpr and tpr:
                color, linestyle = get_method_style(result_file.name)
                plt.plot(
                    fpr,
                    tpr,
                    label=result_file.name,
                    linewidth=3,
                    color=color,
                    linestyle=linestyle,
                )
            else:
                print(
                    f"Warning: No data found for {result_file.name} on dataset {dataset_name}"
                )
        else:
            print(f"Warning: Dataset {dataset_name} not found in {result_file.name}")
            print(f"Datasets found: {dataset_roc_data.keys()}")

    if show_legend:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Keep ticks but remove labels
    plt.tick_params(axis="both", which="major", labelbottom=False, labelleft=False)

    plt.tight_layout()
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    results = [
        # Prompted baselines
        ROCCurveFile(
            name="Gemma-27b (prompted)",
            type="prompted_baseline",
            path=RESULTS_DIR
            / "prompting_baselines_test_performance/baseline_gemma-27b.jsonl",
        ),
        ROCCurveFile(
            name="Llama-70b (prompted)",
            type="prompted_baseline",
            path=RESULTS_DIR
            / "prompting_baselines_test_performance/baseline_llama-70b.jsonl",
        ),
        # Finetuned baselines - using v2 folder for 1B models and 8B
        ROCCurveFile(
            name="Gemma-1b (finetuned)",
            type="finetuned_baseline",
            path=RESULTS_DIR
            / "finetuning_baselines_test_performance_v2/finetuning_gemma_1b_test_optimized_2.jsonl",
        ),
        # ROCCurveFile(
        #     name="Llama-1b (finetuned)",
        #     type="finetuned_baseline",
        #     path=RESULTS_DIR
        #     / "finetuning_baselines_test_performance_v2/finetuning_llama_1b_test_optimized_2.jsonl",
        # ),
        ROCCurveFile(
            name="Llama-8b (finetuned)",
            type="finetuned_baseline",
            path=RESULTS_DIR
            / "finetuning_baselines_test_performance_v2/finetuning_llama_8b_test_optimized_2.jsonl",
        ),
        # ROCCurveFile(
        #     name="Gemma-12b (finetuned)",
        #     type="finetuned_baseline",
        #     path=RESULTS_DIR
        #     / "finetuning_baselines_test_performance/finetuning_gemma_12b_test.jsonl",
        # ),
        ROCCurveFile(
            name="Attention Probe",
            type="probe",
            path=RESULTS_DIR / "evaluate_probes/results_attention_test_1.jsonl",
        ),
        ROCCurveFile(
            name="Softmax Probe",
            type="probe",
            path=RESULTS_DIR / "evaluate_probes/results_softmax_test_1.jsonl",
        ),
    ]

    # Plot ROC curves for different datasets using proper capitalized names
    plot_roc_curves(
        results,
        "Anthropic",
        RESULTS_DIR / "plots/roc_curve_anthropic_with_legend.png",
        show_legend=True,
    )

    plot_roc_curves(
        results,
        "Anthropic",
        RESULTS_DIR / "plots/roc_curve_anthropic_without_legend.png",
        show_legend=False,
    )
