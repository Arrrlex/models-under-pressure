import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from models_under_pressure.config import DATA_DIR, PLOTS_DIR, PROJECT_ROOT
from models_under_pressure.eval_datasets.label_dataset import (
    LabelledDataset,
    label_dataset,
)
from models_under_pressure.interfaces.dataset import Dataset, Label, Message


def gen_llama_rollouts() -> Dict[str, Dataset]:
    """
    Loads all JSON files containing 'llama-70b-3.3' in their names from the rollouts directory.
    Returns a dictionary mapping from the experiment name (e.g. 'insider_trading__onpolicy') to a Dataset object.
    Each Dataset contains:
    - inputs: List of conversations (each conversation is a list of Message objects)
    - ids: List of unique identifiers for each conversation
    - other_fields: Dictionary containing 'is_deceptive' field from graded_deception
    """
    rollouts_dir = PROJECT_ROOT.parent / "deception-detection" / "data" / "rollouts"
    result = {}

    for file_path in rollouts_dir.glob("*llama-70b-3.3.json"):
        print(file_path)
        name = file_path.stem.replace("__llama-70b-3.3", "")

        with open(file_path, "r") as f:
            data = json.load(f)

            # Process each rollout into a record
            inputs = []
            ids = []
            is_deceptive = []

            # Handle different file formats based on naming patterns
            if (
                name.startswith("ai_audit")
                or name.startswith("roleplaying")
                or name.startswith("alpaca")
                or name.startswith("ai_liar")
            ):
                # Process ai_audit, roleplaying, alpaca, and ai_liar formats
                for rollout_idx, rollout in enumerate(data["rollouts"]):
                    input_messages = [
                        Message(role=m["role"], content=m["content"])
                        for m in rollout["input_messages"]
                    ]

                    # Create a separate row for each output message
                    for output_idx, output_msg in enumerate(rollout["output_messages"]):
                        # Combine input messages with this output message
                        messages = input_messages + [
                            Message(
                                role=output_msg["role"], content=output_msg["content"]
                            )
                        ]
                        inputs.append(messages)
                        ids.append(f"{name}_{rollout_idx}_{output_idx}")
                        is_deceptive.append(rollout["graded_deception"][output_idx])
            elif name.startswith("sandbagging"):
                # Process sandbagging format (same as above but remove last input message)
                for rollout_idx, rollout in enumerate(data["rollouts"]):
                    # Remove the last input message
                    input_messages = [
                        Message(role=m["role"], content=m["content"])
                        for m in rollout["input_messages"][:-1]
                    ]

                    # Create a separate row for each output message
                    for output_idx, output_msg in enumerate(rollout["output_messages"]):
                        # Combine input messages with this output message
                        messages = input_messages + [
                            Message(
                                role=output_msg["role"], content=output_msg["content"]
                            )
                        ]
                        inputs.append(messages)
                        ids.append(f"{name}_{rollout_idx}_{output_idx}")
                        is_deceptive.append(rollout["graded_deception"][output_idx])
            elif name.startswith("insider_trading"):
                # TODO: Add processing for insider trading format
                continue
            else:
                print(f"Skipping unknown format: {name}")
                continue

            # Create the Dataset
            dataset = Dataset(
                inputs=inputs, ids=ids, other_fields={"is_deceptive": is_deceptive}
            )

            # Label the dataset
            labelled_dataset = label_dataset(dataset, max_concurrent=500)
            result[name] = labelled_dataset

    return result


def save_rollouts(rollouts: Dict[str, Dataset]) -> None:
    """
    Saves each rollout dataset to data/evals/ais/{name}.jsonl
    """
    ais_dir = DATA_DIR / "evals" / "ais"
    ais_dir.mkdir(parents=True, exist_ok=True)

    for name, dataset in rollouts.items():
        output_path = ais_dir / f"{name}.jsonl"
        print(f"Saving {name} to {output_path}")
        dataset.save_to(output_path, overwrite=True)


def plot_deception_by_label(datasets: List[LabelledDataset]) -> Figure:
    """
    Plots a bar chart comparing is_deceptive values across different labels for each dataset.

    Args:
        datasets: List of LabelledDataset instances to plot

    Returns:
        Figure: The matplotlib figure containing the plot
    """
    # Define the desired label order
    label_order = [Label.LOW_STAKES, Label.AMBIGUOUS, Label.HIGH_STAKES]
    all_labels = [str(label) for label in label_order]

    # Prepare data for plotting
    dataset_names = []
    label_means = {str(label): [] for label in label_order}

    for dataset in datasets:
        dataset_names.append(dataset.ids[0])  # Use first ID as name
        # Group by label and calculate mean is_deceptive for each group
        for label in label_order:
            mask = np.array([lab == label for lab in dataset.labels])

            # Handle both is_deceptive and is_honest fields
            if "is_deceptive" in dataset.other_fields:
                is_deceptive = np.array(
                    [
                        1 if x == "deceptive" else 0
                        for x in dataset.other_fields["is_deceptive"]
                    ]
                )
            elif "is_honest" in dataset.other_fields:
                # Invert is_honest to get is_deceptive
                is_deceptive = np.array(
                    [
                        0 if x == "honest" else 1
                        for x in dataset.other_fields["is_honest"]
                    ]
                )
            else:
                raise ValueError(
                    f"Dataset {dataset.ids[0]} has neither is_deceptive nor is_honest field"
                )

            mean_deception = np.mean(is_deceptive[mask])
            label_means[str(label)].append(mean_deception)

    # Set up the plot
    x = np.arange(len(dataset_names))
    width = 0.25  # Width of each bar
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each label in the specified order
    for label in all_labels:
        offset = width * multiplier
        ax.bar(x + offset, label_means[label], width, label=label)
        multiplier += 1

    # Customize the plot
    ax.set_ylabel("Mean is_deceptive")
    ax.set_title("Mean Deception by Label and Dataset")
    ax.set_xticks(x + width * (len(all_labels) - 1) / 2)
    ax.set_xticklabels(dataset_names, rotation=45, ha="right")
    ax.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(PLOTS_DIR / "deception_by_label.png", bbox_inches="tight", dpi=300)

    return fig


def load_llama_rollouts() -> List[LabelledDataset]:
    """
    Loads all labelled datasets from data/evals/ais/*.jsonl files.

    Returns:
        List[LabelledDataset]: List of loaded labelled datasets
    """
    ais_dir = DATA_DIR / "evals" / "ais"
    datasets = []

    for file_path in ais_dir.glob("*.jsonl"):
        dataset = LabelledDataset.load_from(file_path)
        print(f"Loaded dataset from {file_path.name}")
        print(f"Available fields: {dataset.other_fields.keys()}")
        datasets.append(dataset)

    return datasets


if __name__ == "__main__":
    # rollouts = gen_llama_rollouts()
    # save_rollouts(rollouts)
    datasets = load_llama_rollouts()
    fig = plot_deception_by_label(datasets)
    plt.show()
