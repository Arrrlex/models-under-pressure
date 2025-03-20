import io

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


def parse_conversation(row: pd.Series) -> list[Message]:
    """Parse a conversation from a section text and dialogue text.

    Not currently using this."""

    messages = []
    messages.append(Message(role="system", content=f"Context: {row['section_text']}"))

    for line in row["dialogue"].split("\n"):
        if line.strip():
            role, content = line.split(": ", 1)
            new_role = {
                "Doctor": "assistant",
                "Patient": "user",
                "Guest_family": "user",
            }[role.strip()]
            messages.append(Message(role=new_role, content=content.strip()))

    return messages


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

    df["inputs"] = "Context: " + df["section_text"] + "\n" + df["dialogue"]

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

    df = df.rename(columns={"ID": "original_id"})
    df["ids"] = "test_" + df["original_id"].astype(str)

    df["inputs"] = "Context: " + df["section_text"] + "\n" + df["dialogue"]

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
        action="store_false",
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
