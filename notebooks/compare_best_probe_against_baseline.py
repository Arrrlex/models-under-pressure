# %%
from typing import List

import pandas as pd
from pydantic import ValidationError

from models_under_pressure.config import (
    BASELINE_RESULTS_FILE,
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.results import (
    BaselineResults,
    ContinuationBaselineResults,
    EvaluationResult,
)

probe_results_file = "results_0UAqFzWs.jsonl"
probe_results_path = EVALUATE_PROBES_DIR / probe_results_file

probe_results = []
with open(probe_results_path) as f:
    for line in f:
        if line.strip():
            try:
                probe_results.append(EvaluationResult.model_validate_json(line))
            except ValidationError:
                print(f"Error validating line: {line}")

baseline_results = []
with open(BASELINE_RESULTS_FILE) as f:
    for line in f:
        if line.strip():
            baseline_results.append(
                ContinuationBaselineResults.model_validate_json(line)
            )

print(f"Probe results: {len(probe_results)}")
print(f"Baseline results: {len(baseline_results)}")

# Display for which methods and datasets we have baseline results
for baseline_result in baseline_results:
    print(f"{baseline_result.model_name} - {baseline_result.dataset_name}")


# %%
def plot_probe_vs_baseline_accuracy(
    probe_results: List[EvaluationResult],
    baseline_results: List[BaselineResults],
    valid_responses_only: bool = False,
):
    """Create a bar plot comparing accuracy of probe vs baseline models across datasets.

    Args:
        probe_results: List of EvaluationResult objects containing probe performance
        baseline_results: List of BaselineResults objects containing baseline model performance
        valid_responses_only: If True, only include valid responses when calculating baseline accuracy
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract probe accuracies per dataset
    probe_data = {
        result.dataset_name: result.metrics.metrics["accuracy"]
        for result in probe_results
    }

    # Extract baseline accuracies per dataset and model
    baseline_data = {}
    for result in baseline_results:
        if hasattr(result, "valid_response") and valid_responses_only:
            # Filter for valid responses only
            valid_indices = [
                i for i, valid in enumerate(result.valid_response) if valid
            ]
            if valid_indices:
                correct = sum(
                    result.labels[i] == result.ground_truth[i] for i in valid_indices
                )
                accuracy = correct / len(valid_indices)
                baseline_data[(result.dataset_name, result.model_name)] = accuracy
        else:
            # Use overall accuracy if not filtering or if valid_response not available
            baseline_data[(result.dataset_name, result.model_name)] = result.accuracy

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


_ = plot_probe_vs_baseline_accuracy(
    probe_results, baseline_results, valid_responses_only=True
)


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
