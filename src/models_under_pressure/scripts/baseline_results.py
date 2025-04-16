from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from models_under_pressure.baselines.continuation import likelihood_continuation_prompts
from models_under_pressure.config import DATA_DIR
from models_under_pressure.probes.metrics import tpr_at_fixed_fpr_score
from models_under_pressure.interfaces.results import (
    ContinuationPrompt,
    LikelihoodBaselineResults,
)


def identify_config_name(config: ContinuationPrompt) -> str:
    for key, value in likelihood_continuation_prompts.items():
        if value == config:
            return key
    raise ValueError(f"No matching config found for {config}")


def print_baseline_results(results_file: Path, fpr: float = 0.2, metric: str = "tpr"):
    """Read results and print metrics for each entry in a DataFrame format.

    Args:
        results_file: Path to results file
        fpr: False positive rate threshold for TPR computation
        metric: Either 'tpr' or 'auroc' to compute different metrics
    """
    results = []
    # Read the JSONL file line by line
    with open(results_file) as f:
        for line in f:
            result = LikelihoodBaselineResults.model_validate_json(line)
            results.append(result)

    df_data = []

    for result in results:
        y_true = result.ground_truth
        scores = result.high_stakes_scores

        # print(y_true, scores)

        if metric == "tpr":
            score = tpr_at_fixed_fpr_score(y_true, scores, fpr=fpr)
        elif metric == "auroc":
            score = roc_auc_score(y_true, scores)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        model_prompt = (
            f"{result.model_name} ({identify_config_name(result.prompt_config)})"
        )
        df_data.append(
            {
                "model_prompt": model_prompt,
                "dataset": result.dataset_name,
                "score": score,
            }
        )

    # Create DataFrame and pivot it
    df = pd.DataFrame(df_data)
    df = df.groupby(["model_prompt", "dataset"])["score"].mean().reset_index()
    pivot_df = df.pivot(index="model_prompt", columns="dataset", values="score")

    # Add average column
    pivot_df["Average"] = pivot_df.mean(axis=1)

    metric_name = "TPR" if metric == "tpr" else "AUROC"
    if metric == "tpr":
        print(f"\n{metric_name} at {fpr * 100:.0f}% FPR for each model and dataset:")
    else:
        print(f"\n{metric_name} for each model and dataset:")
    print("-" * 40)
    print(pivot_df.round(3))

    return pivot_df


if __name__ == "__main__":
    results_file = DATA_DIR / "probes/continuation_baseline_results_variations.jsonl"
    print_baseline_results(results_file, fpr=0.01, metric="tpr")
    # print_baseline_results(results_file, fpr=0.01, metric="auroc")
