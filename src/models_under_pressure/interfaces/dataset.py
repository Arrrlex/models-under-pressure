import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel

from models_under_pressure.config import ANTHROPIC_SAMPLES_CSV


class Message(BaseModel):
    role: str
    content: str


class Label(Enum):
    HIGH_STAKES = "high-stakes"
    LOW_STAKES = "low-stakes"
    AMBIGUOUS = "ambiguous"


Input = str | list[Message]


class Dataset(BaseModel):
    inputs: Sequence[Input]
    labels: Sequence[Label]

    ids: Sequence[str]

    other_fields: Mapping[str, Sequence[Any]]

    def to_pandas(self) -> pd.DataFrame:
        base_data = {
            "inputs": self.inputs,
            "labels": self.labels,
            "ids": self.ids,
        }
        # Add each field from other_fields as a separate column
        base_data.update(self.other_fields)
        return pd.DataFrame(base_data)


def load_anthropic_csv(filename: Path) -> Dataset:
    df = pd.read_csv(filename)

    messages = df["messages"].apply(json.loads)

    # Convert high_stakes column to Label enum
    labels = df["high_stakes"].apply(
        lambda x: Label.HIGH_STAKES
        if x == 1
        else Label.LOW_STAKES
        if x == 0
        else Label.AMBIGUOUS  # type: ignore
    )

    # Get IDs
    ids = df["id"].tolist()

    # Get any remaining columns as other fields
    other_fields = {
        col: df[col].tolist()
        for col in df.columns
        if col not in ["messages", "high_stakes", "id"]
    }

    return Dataset(
        inputs=messages.tolist(),
        labels=labels.tolist(),
        ids=ids,
        other_fields=other_fields,
    )


if __name__ == "__main__":
    dataset = load_anthropic_csv(ANTHROPIC_SAMPLES_CSV)
    print(dataset.to_pandas())
