# %%
from typing import List

import numpy as np
import pandas as pd
from pydantic import ValidationError

from models_under_pressure.config import (
    BASELINE_RESULTS_FILE_TEST,
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.results import (
    BaselineResults,
    ContinuationBaselineResults,
    EvaluationResult,
    LikelihoodBaselineResults,
)

probe_results_file = "results_best_probe_test.jsonl"
baseline_results_file = BASELINE_RESULTS_FILE_TEST
probe_results_path = EVALUATE_PROBES_DIR / probe_results_file

probe_results = []
with open(probe_results_path) as f:
    for line in f:
        if line.strip():
            try:
                probe_results.append(EvaluationResult.model_validate_json(line))
            except ValidationError as e:
                print(f"Error validating line: {line}")
                print(f"Error details: {e}")
                # Try to load with strict validation disabled
                try:
                    probe_results.append(
                        EvaluationResult.model_validate_json(line, strict=False)
                    )
                except Exception as e2:
                    print(f"Could not load even with strict=False: {e2}")

baseline_results = []
with open(baseline_results_file) as f:
    for line in f:
        if line.strip():
            result = LikelihoodBaselineResults.model_validate_json(line)
            if result.max_samples is None:
                baseline_results.append(result)

print(f"Probe results: {len(probe_results)}")
print(f"Baseline results: {len(baseline_results)}")

# Display for which methods and datasets we have baseline results
for baseline_result in baseline_results:
    print(f"{baseline_result.model_name} - {baseline_result.dataset_name}")


# %%


def get_readable_model_name(model_name: str) -> str:
    """Convert model names to a more readable format.

    Examples:
        "google/gemma-3-12b-it" -> "Gemma 12B"
        "meta-llama/Llama-3.2-1B-Instruct" -> "Llama 1B"

    Args:
        model_name: The original model name

    Returns:
        A more readable version of the model name
    """
    import re

    # Handle Gemma models
    gemma_match = re.match(r"google/gemma-3-(\d+)b-it", model_name)
    if gemma_match:
        size = gemma_match.group(1)
        return f"Gemma {size}B"

    # Handle Llama models
    llama_match = re.match(r"meta-llama/Llama-3\.[\d]-(\d+)B-Instruct", model_name)
    if llama_match:
        size = llama_match.group(1)
        return f"Llama {size}B"

    if model_name.startswith("Probe"):
        return "Probe"

    # Return original name if no matches
    return model_name


# Add this near the top of the file with other constants
DATASET_NAME_MAPPING = {
    # "mt": "Medical MT Samples",
    "mt": "MT Samples",
    "manual": "Manual",
    "mts": "MTS Dialog",
    "toolace": "ToolACE",
    "anthropic": "Anthropic HH",
    "mental_health": "Mental Health",
    "redteaming": "Aya Red Teaming",
}


def plot_probe_vs_baseline_auroc(
    probe_results: List[EvaluationResult],
    baseline_results: List[LikelihoodBaselineResults],
    output_path: str,
    use_title: bool = True,
):
    """Create a bar plot comparing AUROC scores of probe vs baseline models across datasets.

    Args:
        probe_results: List of EvaluationResult objects containing probe performance
        baseline_results: List of LikelihoodBaselineResults objects containing baseline model performance
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    # Extract probe AUROC per dataset
    probe_data = {
        result.dataset_name: {
            "auroc": result.metrics.metrics["auroc"],
            "model_name": result.config.model_name,
        }
        for result in probe_results
    }

    # Calculate baseline AUROC per dataset and model
    baseline_data = {}
    for result in baseline_results:
        auroc = roc_auc_score(result.ground_truth, result.high_stakes_scores)
        baseline_data[(result.dataset_name, result.model_name)] = auroc

    # Create dataframe for plotting
    plot_data = []
    for dataset in probe_data.keys():
        plot_data.append(
            {
                "Dataset": dataset,
                "Method": f"Probe ({get_readable_model_name(probe_data[dataset]['model_name'])})",
                "AUROC": probe_data[dataset]["auroc"],
            }
        )

        # Group models by provider
        model_groups = {}
        for (ds, model), auroc in baseline_data.items():
            if ds != dataset:
                continue

            # Get provider from part before "/" in model name
            provider = model.split("/")[0] if "/" in model else "other"
            if provider not in model_groups:
                model_groups[provider] = []
            model_groups[provider].append((model, auroc))

        # Sort each group by model size
        for provider in model_groups:

            def get_model_size(model_name: str) -> float:
                import re

                match = re.search(r"-(\d+)[bB]", model_name)
                if match:
                    return int(match.group(1))
                return 0

            model_groups[provider].sort(key=lambda x: get_model_size(x[0]))

        # Add sorted models to plot data
        for provider in model_groups:
            for model, auroc in model_groups[provider]:
                plot_data.append({"Dataset": dataset, "Method": model, "AUROC": auroc})

    df = pd.DataFrame(plot_data)

    # Create color mapping for model families
    providers = {
        model.split("/")[0] if "/" in model else "other"
        for model in df["Method"].unique()
        if not model.startswith("Probe")
    }

    # Create color palette - one base color per provider

    # Simpler pattern set: diagonal right, diagonal left, cross-hatch
    patterns = ["//", "\\\\", "xx"]

    # Create base colors for each provider
    base_colors = {provider: plt.cm.Set3(i) for i, provider in enumerate(providers)}

    # Create color map and pattern map
    color_map = {}
    pattern_map = {}

    # Add probe pattern mapping first
    probe_method = [m for m in df["Method"].unique() if m.startswith("Probe")][0]
    pattern_map[probe_method] = "x"
    color_map[probe_method] = "black"

    for provider in providers:
        provider_models = [
            m for m in df["Method"].unique() if m != probe_method and (provider in m)
        ]
        provider_models = list(set(provider_models))
        # Sort models by size to ensure consistent pattern assignment
        provider_models.sort(key=lambda x: get_model_size(x))

        for i, model in enumerate(provider_models):
            # Same color for all models from same provider
            color_map[model] = base_colors[provider]
            # Different pattern based on model size
            pattern_map[model] = patterns[i % len(patterns)]

    # Pivot the dataframe for plotting
    df_pivot = df.pivot(index="Dataset", columns="Method", values="AUROC")

    # Plot setup
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort methods by provider and size, keeping Probe first
    probe_method = next(m for m in df_pivot.columns if m.startswith("Probe"))
    other_methods = [m for m in df_pivot.columns if not m.startswith("Probe")]

    # Group methods by provider
    provider_groups = {}
    for method in other_methods:
        provider = method.split("/")[0] if "/" in method else "other"
        if provider not in provider_groups:
            provider_groups[provider] = []
        provider_groups[provider].append(method)

    # Sort methods within each provider group by size
    sorted_methods = []
    for provider in sorted(provider_groups.keys()):
        provider_methods = provider_groups[provider]
        provider_methods.sort(key=get_model_size)
        sorted_methods.extend(provider_methods)

    methods = [probe_method] + sorted_methods

    datasets = df_pivot.index.tolist()
    n_methods = len(methods)
    x = np.arange(len(datasets))
    bar_width = 0.8 / n_methods  # total width is 0.8, divided by number of methods

    # Create bars manually with precise control over hatches and colors
    for i, method in enumerate(methods):
        positions = x - 0.4 + (i + 0.5) * bar_width
        _ = ax.bar(
            positions,
            df_pivot[method],
            bar_width,
            label=get_readable_model_name(method),
            color=color_map[method],
            edgecolor="black",
            hatch=pattern_map[method],
            linewidth=1,
        )

    # Increase default font sizes
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "legend.title_fontsize": 14,
        }
    )

    # Update x-axis labels with mapped names - remove rotation and adjust alignment
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_NAME_MAPPING.get(d, d) for d in datasets], rotation=0, ha="center"
    )

    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUROC")
    if use_title:
        ax.set_title(
            f"Probe ({probe_results[0].config.model_name}) vs Baseline Model AUROC by Dataset"
        )

    # Legend outside plot
    ax.legend(
        title="Method",
        # bbox_to_anchor=(1.05, 1),
        # loc="upper left",
        ncol=2,
        loc="lower left",
        borderaxespad=0.4,
        framealpha=1.0,
    )

    # Adjust layout - we can reduce bottom margin since labels are horizontal now
    plt.tight_layout()
    plt.subplots_adjust(
        right=0.85, bottom=0.15
    )  # Reduced bottom margin from 0.2 to 0.15
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path)
    plt.show()


output_path = "../data/plots/probe_vs_baseline_auroc_test.pdf"
_ = plot_probe_vs_baseline_auroc(
    probe_results, baseline_results, output_path, use_title=False
)

# %%


# NOTE STUFF BELOW IS FOR ANALYZING CONTINUATION BASELINE RESULTS (NOT LIKELIHOOD BASELINE)


def plot_probe_vs_baseline_accuracy(
    probe_results: List[EvaluationResult], baseline_results: List[BaselineResults]
):
    """Create a bar plot comparing accuracy of probe vs baseline models across datasets.

    Args:
        probe_results: List of EvaluationResult objects containing probe performance
        baseline_results: List of BaselineResults objects containing baseline model performance
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract probe accuracies per dataset
    probe_data = {
        result.dataset_name: result.metrics.metrics["accuracy"]
        for result in probe_results
    }

    # Extract baseline accuracies per dataset and model
    baseline_data = {
        (result.dataset_name, result.model_name): result.accuracy
        for result in baseline_results
    }

    # Create dataframe for plotting
    plot_data = []
    for dataset in probe_data.keys():
        plot_data.append(
            {"Dataset": dataset, "Method": "Probe", "Accuracy": probe_data[dataset]}
        )

        # Group models by provider
        model_groups = {}

        for (ds, model), acc in baseline_data.items():
            if ds != dataset:
                continue

            # Get provider from part before "/" in model name
            provider = model.split("/")[0] if "/" in model else "other"
            if provider not in model_groups:
                model_groups[provider] = []
            model_groups[provider].append((model, acc))

        # Sort each group by model size
        for provider in model_groups:

            def get_model_size(model_name: str) -> float:
                import re

                # Extract size from model name (e.g. "7" from "model-7b")
                match = re.search(r"-(\d+)[bB]", model_name)
                if match:
                    return int(match.group(1))
                return 0

            model_groups[provider].sort(key=lambda x: get_model_size(x[0]))
        # Add sorted models to plot data
        for provider in model_groups:
            for model, acc in model_groups[provider]:
                plot_data.append({"Dataset": dataset, "Method": model, "Accuracy": acc})

    df = pd.DataFrame(plot_data)

    # Create grouped bar plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot bars grouped by dataset, maintaining order from plot_data
    methods = df["Method"].unique()  # Get methods in order they appear in plot_data
    df_pivot = df.pivot(index="Dataset", columns="Method", values="Accuracy")
    df_pivot = df_pivot[methods]  # Reorder columns to match plot_data order
    df_pivot.plot(kind="bar", ax=ax)

    plt.title("Probe vs Baseline Model Accuracy by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.legend(title="Method")
    plt.tight_layout()

    return plt.gcf()


_ = plot_probe_vs_baseline_accuracy(probe_results, baseline_results)


# %%
# Also show how often the baselines returned invalid responses
def get_invalid_response_table(
    baseline_results: List[ContinuationBaselineResults],
) -> pd.DataFrame:
    """Print a table showing invalid response rates for each model and dataset.

    Args:
        baseline_results: List of ContinuationBaselineResults
    """
    import re

    # Group results by model and dataset
    model_groups = {}
    for result in baseline_results:
        model = result.model_name
        # Get provider from part before "/" in model name
        provider = model.split("/")[0] if "/" in model else "other"
        if provider not in model_groups:
            model_groups[provider] = {}
        if model not in model_groups[provider]:
            model_groups[provider][model] = {}

        invalid_pct = (
            sum(not v for v in result.valid_response) / len(result.valid_response) * 100
        )
        model_groups[provider][model][result.dataset_name] = invalid_pct

    # Create list of rows for DataFrame
    rows = []
    for provider in model_groups:
        # Sort models by size within provider
        sorted_models = sorted(
            model_groups[provider].keys(),
            key=lambda x: int(re.search(r"-(\d+)[bB]", x).group(1))
            if re.search(r"-(\d+)[bB]", x)
            else 0,
        )

        for model in sorted_models:
            row = {"Provider": provider, "Model": model}
            row.update(model_groups[provider][model])
            rows.append(row)

    # Create DataFrame and display
    df = pd.DataFrame(rows)
    df = df.set_index(["Provider", "Model"])

    pd.options.display.float_format = "{:.1f}%".format
    return df


print("Invalid Response Rates:")
df = get_invalid_response_table(baseline_results)
df

# %%


def print_sample_responses(
    baseline_results: List[ContinuationBaselineResults],
    model_name: str,
    dataset_name: str,
    k: int = 5,
):
    """Print k sample responses for a specific model and dataset.

    Args:
        baseline_results: List of ContinuationBaselineResults
        model_name: Name of the model to show responses for
        dataset_name: Name of the dataset to show responses for
        k: Number of samples to show (default: 5)
    """
    # Find the matching result
    result = next(
        (
            r
            for r in baseline_results
            if r.model_name == model_name and r.dataset_name == dataset_name
        ),
        None,
    )

    if not result:
        print(f"No results found for model '{model_name}' on dataset '{dataset_name}'")
        return

    print(f"\nSample responses from {model_name} on {dataset_name}:\n")

    # Get k random indices while preserving order
    import random

    # Load the dataset to get input texts
    dataset_path = EVAL_DATASETS[dataset_name]
    dataset = LabelledDataset.load_from(dataset_path)
    # with open(dataset_path) as f:
    #     dataset = (
    #         [json.loads(line) for line in f]
    #         if str(dataset_path).endswith(".jsonl")
    #         else pd.read_csv(dataset_path).to_dict("records")
    #     )

    # Create mapping from id to input text
    id_to_input = {
        sample_id: sample_input
        for sample_id, sample_input in zip(dataset.ids, dataset.inputs)
    }

    indices = sorted(random.sample(range(len(result.ids)), min(k, len(result.ids))))

    for idx in indices:
        input_id = result.ids[idx]
        ground_truth = result.ground_truth[idx]
        prediction = result.labels[idx]
        full_response = result.full_response[idx]
        valid = result.valid_response[idx]

        # Truncate long texts for display
        def truncate(text: str, max_len: int = 100) -> str:
            if len(text) > max_len:
                return text[:max_len] + "..."
            return text

        print(f"Sample {idx}:")
        print(f"Input ID: {input_id}")
        print(f"Input text: {truncate(id_to_input[input_id])}")
        print(f"Ground truth: {ground_truth}")
        print(f"Prediction: {prediction}")
        print(f"Valid response: {valid}")
        print(f"Full response: {full_response}")
        print("-" * 80)


# Example usage:
# print_sample_responses(baseline_results, "google/gemma-3-1b-it", "mt", k=10)
print_sample_responses(baseline_results, "google/gemma-3-12b-it", "manual", k=10)

# %%
