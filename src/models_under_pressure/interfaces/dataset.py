from enum import Enum
from typing import Any, Callable, Mapping, Self, Sequence, overload

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


class Record(BaseModel):
    input: Input
    label: Label
    id: str
    other_fields: Mapping[str, Any]


class Dataset(BaseModel):
    inputs: Sequence[Input]
    labels: Sequence[Label]
    ids: Sequence[str]
    other_fields: Mapping[str, Sequence[Any]]

    def records(self) -> Sequence[Record]:
        return [
            Record(
                input=input,
                label=label,
                id=id,
                other_fields={k: v[i] for k, v in self.other_fields.items()},
            )
            for i, (input, label, id) in enumerate(
                zip(self.inputs, self.labels, self.ids)
            )
        ]

    @classmethod
    def from_records(cls, records: Sequence[Record]) -> Self:
        field_keys = records[0].other_fields.keys()
        return cls(
            inputs=[r.input for r in records],
            labels=[r.label for r in records],
            ids=[r.id for r in records],
            other_fields={k: [r.other_fields[k] for r in records] for k in field_keys},
        )

    def to_pandas(self) -> pd.DataFrame:
        base_data = {
            "inputs": self.inputs,
            "labels": self.labels,
            "ids": self.ids,
        }
        # Add each field from other_fields as a separate column
        base_data.update(self.other_fields)
        return pd.DataFrame(base_data)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Self:
        return cls.from_records(
            [
                Record(
                    input=row["inputs"],
                    label=row["labels"],
                    id=row["ids"],
                    other_fields=row.drop(
                        columns=["inputs", "labels", "ids"]
                    ).to_dict(),
                )
                for _, row in df.iterrows()
            ]
        )

    def labels_numpy(self) -> Float[np.ndarray, " batch_size"]:
        return np.array([label.to_int() for label in self.labels])

    @overload
    def __getitem__(self, idx: int) -> Record: ...

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    def __getitem__(self, idx: int | slice) -> Self | Record:
        if isinstance(idx, slice):
            return type(self)(
                inputs=self.inputs[idx],
                labels=self.labels[idx],
                ids=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
        else:
            return Record(
                input=self.inputs[idx],
                label=self.labels[idx],
                id=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )

    def filter(self, filter_fn: Callable[[Record], bool]) -> Self:
        return type(self).from_records([r for r in self.records() if filter_fn(r)])
