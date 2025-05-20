import glob
import json
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_under_pressure.config import RESULTS_DIR
from models_under_pressure.experiments.evaluate_probes import calculate_metrics
from models_under_pressure.experiments.monitoring_cascade import (
    get_abbreviated_model_name,
)

# Define colors from monitoring_cascade.py
MODEL_COLORS = {
    "gemma-1b": {
        "baseline": "#5555FF",
    },
    "gemma-12b": {
        "baseline": (0.0, 0.807843137254902, 0.8196078431372549),
    },
    "gemma-27b": {
        "baseline": (0.0, 0.654902, 0.660784),
    },
    "llama-1b": {
        "baseline": "#9999FF",
    },
    "llama-8b": {
        "baseline": "#0000FF",
    },
    "llama-70b": {
        "baseline": (0.0, 0.5019607843137255, 0.5019607843137255),
    },
}

# Define patterns for different prompts
PROMPT_PATTERNS = {
    "default": "",
    "single_word": "///",
    "prompt_at_end": "\\\\\\",
    "single_letter": "xxx",
}

# Define model size order (smallest to largest)
MODEL_SIZE_ORDER = {
    "llama-1b": 0,
    "llama-3b": 1,
    "llama-8b": 2,
    "llama-70b": 3,
    "gemma-1b": 4,
    "gemma-12b": 5,
    "gemma-27b": 6,
}


def get_prompt_name(prompt_config: Optional[Dict[str, Any]]) -> str:
    """Get prompt name by matching content with likelihood_continuation_prompts."""
    if not prompt_config:
        return "default"

    # Extract key components for matching
    high_stakes = prompt_config.get("high_stakes_completion", "")
    low_stakes = prompt_config.get("low_stakes_completion", "")
    system_prompt = prompt_config.get("system_prompt", "")
    user_prompt = prompt_config.get("user_prompt", "")

    # Match based on completion patterns
    if high_stakes == "high" and low_stakes == "low":
        if system_prompt and "single word" in system_prompt.lower():
            return "single_word"
        elif "CONTEXT" in user_prompt:
            return "prompt_at_end"
        else:
            raise ValueError(f"Unknown prompt: {system_prompt}")
    elif high_stakes == "A" and low_stakes == "B":
        return "single_letter"
    elif high_stakes == "The given conversation is high-stakes.":
        return "default"
    else:
        raise ValueError(f"Unknown prompt: {prompt_config}")


def get_model_sort_key(model_name: str) -> tuple[int, int]:
    """Get sort key for model name: (model_type, size_order)"""
    model_type = 0 if "llama" in model_name.lower() else 1
    short_name = get_abbreviated_model_name(model_name)
    size_order = MODEL_SIZE_ORDER.get(short_name, 999)
    return (model_type, size_order)


INPUT_DIR = RESULTS_DIR / "all_prompts"

# Find all result files
result_files = glob.glob(
    os.path.join(INPUT_DIR, "continuation_baseline_results_*.jsonl")
)

auroc_results = []

for file_path in result_files:
    with open(file_path, "r") as f:
        # Read all lines from the file
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                model_name = data.get("model_name", os.path.basename(file_path))
                prompt_config = data.get("prompt_config")
                prompt_name = get_prompt_name(prompt_config)
                # Compute metrics using calculate_metrics
                metrics = calculate_metrics(
                    np.array(data["ground_truth"]),
                    np.array(data["high_stakes_scores"]),
                    fpr=0.01,
                )
                auroc = metrics["auroc"]
                auroc_results.append(
                    {
                        "model": model_name,
                        "prompt": prompt_name,
                        "file": os.path.basename(file_path),
                        "auroc": auroc,
                    }
                )

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(auroc_results)

# Get unique models and prompts, sorted appropriately
models = sorted(df["model"].unique(), key=get_model_sort_key)
prompts = sorted(df["prompt"].unique())

# Set up the plot
plt.rcParams.update({"font.size": 14})  # Increase base font size
fig, ax = plt.subplots(figsize=(12, 6))

# Add grid
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

# Calculate bar width and positions
n_models = len(models)
n_prompts = len(prompts)
bar_width = 0.8 / n_prompts
x = np.arange(n_models)

print([result["auroc"] for result in auroc_results if "70B" in result["model"]])

# Plot bars for each prompt
for i, prompt in enumerate(prompts):
    prompt_data = df[df["prompt"] == prompt]
    values = []
    for model in models:
        model_data = prompt_data[prompt_data["model"] == model]
        # Calculate mean AUROC over all datasets for this model and prompt
        mean_auroc = model_data["auroc"].mean() if not model_data.empty else None
        values.append(mean_auroc)

    # Calculate position for this prompt's bars
    position = x + bar_width * (i - n_prompts / 2 + 0.5)

    # Get colors for each model
    colors = [
        MODEL_COLORS.get(get_abbreviated_model_name(model), {}).get(
            "baseline", "skyblue"
        )
        for model in models
    ]

    # Plot bars with patterns and outlines
    bars = ax.bar(
        position,
        values,
        width=bar_width,
        label=prompt,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
        hatch=PROMPT_PATTERNS.get(prompt, ""),
    )

    # Remove value labels
    # for bar, value in zip(bars, values):
    #     if value is not None:
    #         ax.annotate(
    #             f"{value:.3f}",
    #             xy=(bar.get_x() + bar.get_width() / 2, value),
    #             xytext=(0, 3),
    #             textcoords="offset points",
    #             ha="center",
    #             va="bottom",
    #         )

# Customize the plot
ax.set_ylabel("AUROC", fontsize=16)
ax.set_ylim(0.4, 1)  # Changed from (0, 1) to (0.5, 1)
ax.set_xticks(x)
ax.set_xticklabels(
    [get_abbreviated_model_name(model) for model in models],
    rotation=45,
    ha="right",
    fontsize=14,
)

# Add legend with simplified style
legend = ax.legend(
    title="Prompt Template",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=14,
    title_fontsize=16,
    frameon=True,
    edgecolor="black",
)

# Remove color from legend markers
for handle in legend.legend_handles:
    handle.set_color("white")
    handle.set_edgecolor("black")

plt.tight_layout()
# plt.show()
plt.savefig("baseline_prompts.pdf", bbox_inches="tight")
