import io

import pandas as pd
import requests

from models_under_pressure.config import (
    EVAL_DATASETS_BALANCED,
    EVAL_DATASETS_RAW,
)
from models_under_pressure.eval_datasets.label_dataset import (
    create_eval_dataset,
)
from models_under_pressure.interfaces.dataset import Dataset, Message

URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/db552cde9da99ff6e24cc6b1b5de5415d83f1850/Main-Dataset/MTS-Dialog-ValidationSet.csv"
# TODO Also use the training set: https://github.com/abachaa/MTS-Dialog/blob/main/Main-Dataset/MTS-Dialog-TrainingSet.csv


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


def load_mts_raw_dataset() -> Dataset:
    print("Downloading MTS dataset")
    response = requests.get(URL)
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))

    df["inputs"] = "Context: " + df["section_text"] + "\n" + df["dialogue"]

    return Dataset.from_pandas(df, field_mapping={"ids": "ID"})


def create_mts_dataset(num_samples: int = 100, recompute: bool = False):
    dataset = load_mts_raw_dataset()
    dataset = dataset.sample(num_samples)
    return create_eval_dataset(
        dataset,
        raw_output_path=EVAL_DATASETS_RAW["mts"],
        balanced_output_path=EVAL_DATASETS_BALANCED["mts"],
        recompute=recompute,
    )


if __name__ == "__main__":
    create_mts_dataset(num_samples=20)
