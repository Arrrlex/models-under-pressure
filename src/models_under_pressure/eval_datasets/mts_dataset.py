import io
import re

import pandas as pd
import requests

from models_under_pressure.config import (
    EVAL_DATASETS_BALANCED,
    EVAL_DATASETS_RAW,
)
from models_under_pressure.eval_datasets.label_dataset import (
    create_eval_dataset,
    create_test_dataset,
)
from models_under_pressure.interfaces.dataset import Dataset, Message

system_prompt_template = """You are a medical assistant, interacting directly with patients and guest families.
Below you find additional context about the current conversation.

## Conctext

{context}"""


def parse_conversation(row: pd.Series, strict: bool = True) -> list[Message] | None:
    """Parse a conversation from a section text and dialogue text.
    Returns None if parsing fails."""

    try:
        messages = []
        messages.append(
            {
                "role": "system",
                "content": system_prompt_template.format(context=row["section_text"]),
            }
        )

        # Match all role-content pairs in the dialogue
        pattern = r"(\S+):\s*((?:(?!\S+:).*\n?)*)"
        matches = re.finditer(pattern, row["dialogue"])

        for match in matches:
            role = match.group(1)
            content = match.group(2).strip()

            if strict and role not in ["Doctor", "Patient"]:
                raise ValueError(
                    "Only roles 'Doctor' and 'Patient' are supported in strict mode!"
                )

            if role.startswith("Doctor"):
                new_role = "assistant"
            elif role.startswith("Patient"):
                new_role = "user"
            elif role.startswith("Guest_family"):
                new_role = "user"
            # TODO This will cause issues as some conversations are between Doctor and Guest clinician
            elif role.startswith("Guest_clinician") or role.startswith(
                "Guest_clinican"
            ):
                new_role = "assistant"
            else:
                raise ValueError(f"Unknown role: {role}")

            messages.append({"role": new_role, "content": content})

        return messages
    except ValueError as e:
        print(f"Warning: Failed to parse conversation: {e}")
        return None


def load_mts_raw_dataset(split: str = "validation") -> Dataset:
    print("Downloading MTS dataset")
    if split == "validation":
        URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/refs/heads/main/Main-Dataset/MTS-Dialog-ValidationSet.csv"
    elif split == "training":
        URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/refs/heads/main/Main-Dataset/MTS-Dialog-TrainingSet.csv"
    else:
        raise ValueError(
            f"Invalid split: {split}. Should be 'validation' or 'training'."
        )
    response = requests.get(URL)
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))

    df = df.rename(columns={"ID": "original_id"})
    df["ids"] = f"{split}_" + df["original_id"].astype(str)

    # Parse conversations and filter out failed parses
    df["inputs"] = df.apply(parse_conversation, axis=1, result_type="reduce")  # type: ignore
    df = df[df["inputs"].notna()]  # Remove rows where parsing failed

    return Dataset.from_pandas(df)


def load_mts_raw_test_dataset() -> Dataset:
    print("Downloading MTS test dataset")
    URL1 = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/refs/heads/main/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"
    URL2 = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/refs/heads/main/Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv"

    response1 = requests.get(URL1)
    response1.raise_for_status()
    df1 = pd.read_csv(io.BytesIO(response1.content))

    response2 = requests.get(URL2)
    response2.raise_for_status()
    df2 = pd.read_csv(io.BytesIO(response2.content))

    df = pd.concat([df1, df2])

    # TODO They use the same IDs for the two files but the files have different items,
    # so create the IDs differently
    df = df.rename(columns={"ID": "original_id"})
    df["ids"] = "test_" + df["original_id"].astype(str)

    # Parse conversations and filter out failed parses
    df["inputs"] = df.apply(parse_conversation, axis=1, result_type="reduce")  # type: ignore
    df = df[df["inputs"].notna()]  # Remove rows where parsing failed

    return Dataset.from_pandas(df)


def create_mts_dataset(
    num_samples: int = 100, recompute: bool = False, split: str | None = None
):
    if split is None:
        val_dataset = load_mts_raw_dataset(split="validation")
        train_dataset = load_mts_raw_dataset(split="training")
        val_records = list(val_dataset.to_records())
        train_records = list(train_dataset.to_records())
        combined_records = val_records + train_records
        dataset = Dataset.from_records(combined_records)  # type: ignore
    elif split == "test":
        dataset = load_mts_raw_test_dataset()
    else:
        dataset = load_mts_raw_dataset(split=split)
    print(f"Loaded {len(dataset)} samples.")

    if len(dataset) > num_samples:
        dataset = dataset.sample(num_samples)

    if split == "test":
        return create_test_dataset(
            dataset,
            dataset_name="mts",
            recompute=recompute,
            remove_dev_samples=False,  # Dataset is split into separate parts already
        )
    else:
        return create_eval_dataset(
            dataset,
            raw_output_path=EVAL_DATASETS_RAW["mts"],
            balanced_output_path=EVAL_DATASETS_BALANCED["mts"],
            recompute=recompute,
        )


def get_mts_samples_by_ids(target_ids: list[str]) -> Dataset:
    """Get MTS samples corresponding to the given list of IDs.

    Args:
        target_ids: List of IDs to match against MTS samples.

    Returns:
        Dataset: A dataset containing all MTS samples that match the given IDs.
    """
    # Convert target IDs to set for efficient lookup
    target_ids_set = set(target_ids)

    # Load all available MTS data
    val_dataset = load_mts_raw_dataset(split="validation")
    train_dataset = load_mts_raw_dataset(split="training")
    test_dataset = load_mts_raw_test_dataset()

    # Combine all datasets
    all_records = (
        list(val_dataset.to_records())
        + list(train_dataset.to_records())
        + list(test_dataset.to_records())
    )
    all_dataset = Dataset.from_records(all_records)  # type: ignore

    # Filter to only include samples with IDs in the target set
    filtered_dataset = all_dataset.filter(lambda r: r.id in target_ids_set)

    if len(filtered_dataset) != len(target_ids):
        print(
            f"Warning: Found {len(filtered_dataset)} samples in MTS dataset, but {len(target_ids)} IDs were provided."
        )

    return filtered_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use",
    )
    parser.add_argument(
        "--recompute",
        type=bool,
        default=False,
        help="Recompute labels even if they already exist",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split to use (train, dev or test). If not provided, train and dev are used.",
    )
    args = parser.parse_args()

    create_mts_dataset(
        num_samples=args.num_samples, recompute=args.recompute, split=args.split
    )
