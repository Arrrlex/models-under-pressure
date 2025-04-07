# %%

import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np

# directory = "/Users/john/Downloads"
# files = ["results_0UAqFzWs_hyperparams.jsonl", "results_DK4FrUMp_hyperparams.jsonl"]
# files = ["results_0UAqFzWs.jsonl", "results_DK4FrUMp.jsonl"]
directory = "../data/results/evaluate_probes/"
files = ["results_colm_test.jsonl"]

data = []
for file in files:
    with open(f"{directory}/{file}", "r") as f:
        data.extend([json.loads(line) for line in f])


# Print config for data1
# Print data1 results grouped by probe config
current_probe = None
current_params = None

for result in data:
    probe = result["config"]["probe_name"]
    params = result["config"]["hyper_params"]
    layer = result["config"]["layer"]

    if probe != current_probe or params != current_params:
        if current_probe is not None:
            print()  # Add spacing between configs

        print("Probe:", probe, ", Layer:", layer)
        print("Hyperparameters:", params)
        print("\nPerformance:")
        current_probe = probe
        current_params = params

    print(f"{result['dataset_name']}: {result['metrics']['metrics']['auroc']:.3f}")

# %%


def plot_best_results(
    data: list[dict],
    output_path: str | None = None,
    name_mapping: dict[str, str] | None = None,
    dataset_mapping: dict[str, str] | None = None,
    exclude_datasets: list[str] | None = None,
    show_values: bool = True,  # New parameter to control value labels
) -> None:
    # Group results by probe_name and dataset
    probe_results = {}
    probe_best_params = {}

    for result in data:
        probe = result["config"]["probe_name"]
        dataset = result["dataset_name"]

        # Skip excluded datasets
        if exclude_datasets and dataset in exclude_datasets:
            continue

        auroc = result["metrics"]["metrics"]["auroc"]
        params = result["config"]["hyper_params"]

        if probe not in probe_results:
            probe_results[probe] = {}
            probe_best_params[probe] = {}

        if dataset not in probe_results[probe] or auroc > probe_results[probe][dataset]:
            probe_results[probe][dataset] = auroc
            probe_best_params[probe][dataset] = params

    # Get unique datasets and probes
    datasets = sorted(
        list(
            {
                dataset
                for probe_dict in probe_results.values()
                for dataset in probe_dict.keys()
            }
        )
    )
    # Map dataset names if mapping is provided
    display_datasets = [
        dataset_mapping.get(d, d) if dataset_mapping else d for d in datasets
    ]
    probes = sorted(list(probe_results.keys()))

    # Set style and colors - colorblind-friendly palette
    colors = [
        "#0077BB",  # Blue
        "#EE7733",  # Orange
        "#009988",  # Teal
        "#CC3311",  # Red-brown
        "#33BBEE",  # Sky blue
        "#EE3377",  # Magenta
    ]

    # Set up the plot with a larger figure and better aspect ratio
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up the plot
    x = np.arange(len(datasets))
    width = 0.8 / len(probes)

    # Plot bars for each probe
    for i, probe in enumerate(probes):
        aurocs = [probe_results[probe].get(dataset, 0) for dataset in datasets]
        offset = width * (i - len(probes) / 2 + 0.5)

        label = name_mapping.get(probe, probe) if name_mapping else probe

        bars = ax.bar(
            x + offset,
            aurocs,
            width,
            label=label,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels only if show_values is True
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                    fontsize=16,
                )

    # Customize the plot
    ax.set_ylabel("AUROC", fontsize=18)
    ax.set_xlabel("Dataset", fontsize=18)  # Added x-axis label
    # ax.set_title(
    #    "AUROC Scores by Probe and Dataset", fontsize=14, fontweight="bold", pad=20
    # )
    ax.set_xticks(x)

    # Wrap long labels into two lines with a maximum width of 12 characters
    wrapped_labels = [textwrap.fill(label, width=12) for label in display_datasets]

    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize=14)

    # Set y-axis tick font size
    ax.tick_params(axis="y", labelsize=16)

    # Customize grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.9)
    ax.set_axisbelow(True)

    # Customize legend with larger fonts
    ax.legend(
        title="Probe Types",
        title_fontsize=18,
        fontsize=16,
        # bbox_to_anchor=(1.02, 1),
        # loc="upper left",
        loc="lower right",
        framealpha=0.9,
    )

    # Set y-axis limits with some padding
    ax.set_ylim(0.5, 1)

    # Print best hyperparameters with better formatting
    print("\nBest hyperparameters for each probe and dataset:")
    print("=" * 50)
    for probe in probes:
        display_name = name_mapping.get(probe, probe) if name_mapping else probe
        print(f"\n📊 {display_name}:")
        for dataset in datasets:
            if dataset in probe_best_params[probe]:
                print(f"  📎 {dataset}:")
                for param, value in probe_best_params[probe][dataset].items():
                    print(f"     • {param}: {value}")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


# Example usage:
name_mapping = {
    "difference_of_means": "Difference of Means",
    "pytorch_per_token_probe": "Per-Token Linear Probe",
    "sklearn_mean_agg_probe": "Per-Sample Linear Probe",
    "lda": "LDA",
}
dataset_mapping = {
    "mt": "MT Samples",
    "manual": "Manual",
    "mts": "MTS Dialog",
    "toolace": "ToolACE",
    "anthropic": "Anthropic HH",
    "mental_health": "Mental Health",
    "redteaming": "Aya Red Teaming",
}
plot_best_results(
    data,
    name_mapping=name_mapping,
    dataset_mapping=dataset_mapping,
    output_path="../data/results/plots/compare_probes_colm.pdf",
    # exclude_datasets=["redteaming", "mental_health"],
    show_values=False,
)
# %%
