from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from jaxtyping import Float
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class Label(Enum):
    LOW_STAKES = "low-stakes"
    HIGH_STAKES = "high-stakes"
    AMBIGUOUS = "ambiguous"

    def to_int(self) -> int:
        return {
            Label.LOW_STAKES: 0,
            Label.HIGH_STAKES: 1,
            Label.AMBIGUOUS: 2,
        }[self]

    @classmethod
    def from_int(cls, i: int) -> "Label":
        return {
            0: cls.LOW_STAKES,
            1: cls.HIGH_STAKES,
            2: cls.AMBIGUOUS,
        }[i]


Dialogue = Sequence[Message]
Input = str | Dialogue


def to_dialogue(input: Input) -> Dialogue:
    if isinstance(input, str):
        return [Message(role="user", content=input)]
    else:
        return input


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

    def labels_numpy(self) -> Float[np.ndarray, " batch_size"]:
        return np.array([label.to_int() for label in self.labels])

    def __getitem__(self, idx: int | slice) -> "Dataset":
        if isinstance(idx, slice):
            return Dataset(
                inputs=self.inputs[idx],
                labels=self.labels[idx],
                ids=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
        else:
            return Dataset(
                inputs=[self.inputs[idx]],
                labels=[self.labels[idx]],
                ids=[self.ids[idx]],
                other_fields={k: [v[idx]] for k, v in self.other_fields.items()},
            )
