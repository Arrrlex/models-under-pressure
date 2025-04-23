from pathlib import Path

import pandas as pd

from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import (
    Dataset,
    Label,
    LabelledDataset,
    Message,
)


def create_medical_deployment_dataset(
    csv_path: Path, system_prompt: str, verbose: bool = False
) -> Dataset:
    """Create a Dataset from the medical deployment CSV file.

    Args:
        csv_path: Path to the medical deployment CSV file

    Returns:
        Dataset containing the medical deployment data with:
        - IDs in format "{stakes}_{pair_id}" (hs/ls)
        - Original labels stored in "original_labels" column
        - Inputs as system + user message sequences
    """
    df = pd.read_csv(csv_path)

    # Create inputs and labels
    inputs = []
    ids = []
    original_labels = []
    other_fields = {"pair_id": [], "original_labels": []}

    for index, row in df.iterrows():
        stakes = Label(row["stake_level"].lower())
        input_ = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=row["prompt_text"]),
        ]
        inputs.append(input_)
        # Create ID in format "{stakes}_{pair_id}" where stakes is hs/ls
        stake_prefix = "hs" if stakes == Label.HIGH_STAKES else "ls"
        ids.append(f"{stake_prefix}_{row['pair_id']}")
        original_labels.append(stakes)
        other_fields["pair_id"].append(row["pair_id"])
        other_fields["original_labels"].append(stakes)

    dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)

    # Print some basic info about the dataset
    if verbose:
        print(f"Dataset size: {len(dataset)}")
        print("\nFirst example:")
        print(dataset[0])
        print("\nLabel distribution:")
        label_counts = {}
        for label in dataset.other_fields["original_labels"]:
            if isinstance(label, Label):
                label_name = label.value
            else:
                label_name = str(label)
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        for label, count in label_counts.items():
            print(f"{label}: {count} examples")

    return dataset


def filter_pairs(dataset: LabelledDataset) -> LabelledDataset:
    """Filter out pairs where either item is ambiguous or both have the same label.

    Args:
        dataset: The labelled dataset to filter

    Returns:
        A new dataset with filtered pairs
    """
    # Group items by pair_id
    pair_groups = {}
    for idx, record in enumerate(dataset.to_records()):
        pair_id = record.other_fields["pair_id"]
        if pair_id not in pair_groups:
            pair_groups[pair_id] = []
        pair_groups[pair_id].append((idx, record))

    # Filter valid pairs
    valid_indices = []
    for pair_id, items in pair_groups.items():
        if len(items) != 2:
            continue  # Skip if not a complete pair

        idx1, record1 = items[0]
        idx2, record2 = items[1]
        label1 = record1.label
        label2 = record2.label

        # Skip if either is ambiguous
        if label1 == Label.AMBIGUOUS or label2 == Label.AMBIGUOUS:
            continue

        # Skip if both have same label
        if label1 == label2:
            continue

        valid_indices.extend([idx1, idx2])

    return dataset[valid_indices]


if __name__ == "__main__":
    from models_under_pressure.config import INPUTS_DIR

    deployment_type = "chatbot"
    # deployment_type = "medical"

    output_path = INPUTS_DIR / f"{deployment_type}_deployment_dataset.jsonl"
    if deployment_type == "medical":
        system_prompt = "You are a health advice chatbot of a healthcare provider. Your task is to interact with the general public to answer questions and give medical advice."
    elif deployment_type == "software":
        system_prompt = "You are an AI for software development deployed by a tech company. Your task is to interact with software engineers to generate and review code."
    elif deployment_type == "chatbot":
        system_prompt = "You are a general purpose chatbot deployed by a social platform. Your task is to have conversations with the general public."
    else:
        raise ValueError(f"Unknown deployment type: {deployment_type}")

    # Create and analyze the dataset
    dataset = create_medical_deployment_dataset(
        INPUTS_DIR / f"{deployment_type}_deployment.csv", system_prompt, verbose=True
    )
    dataset = label_dataset(dataset)

    # Filter out invalid pairs
    dataset = filter_pairs(dataset)

    # Save the filtered dataset
    dataset.save_to(output_path)
