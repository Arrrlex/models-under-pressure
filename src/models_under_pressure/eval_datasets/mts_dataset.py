import io

import pandas as pd
import requests

from models_under_pressure.config import EVAL_DATASETS
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import Dataset, Message

URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/db552cde9da99ff6e24cc6b1b5de5415d83f1850/Main-Dataset/MTS-Dialog-ValidationSet.csv"


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


def load_mts_dataset() -> Dataset:
    print("Downloading MTS dataset")
    response = requests.get(URL)
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))

    df["inputs"] = "Context: " + df["section_text"] + "\n" + df["dialogue"]

    return Dataset.from_pandas(df, field_mapping={"ids": "ID"})


def main(overwrite: bool = False):
    dataset = load_mts_dataset()
    dataset = label_dataset(dataset)
    print(f"High-stakes vs low-stakes: {pd.Series(dataset.labels).value_counts()}")
    dataset.save_to(EVAL_DATASETS["mts"], overwrite=overwrite)


if __name__ == "__main__":
    main(overwrite=True)
