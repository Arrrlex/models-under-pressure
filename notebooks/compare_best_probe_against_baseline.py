# %%
from typing import List

import pandas as pd
from pydantic import ValidationError

from models_under_pressure.config import (
    BASELINE_RESULTS_FILE,
    EVALUATE_PROBES_DIR,
)
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
