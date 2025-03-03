import hashlib
import json
from enum import Enum
from functools import cached_property
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

    @classmethod
    def from_records(cls, records: Sequence[Record]) -> Self:
        field_keys = records[0].other_fields.keys() if len(records) > 0 else []
        return cls(
            inputs=[r.input for r in records],
            labels=[r.label for r in records],
            ids=[r.id for r in records],
            other_fields={k: [r.other_fields[k] for r in records] for k in field_keys},
        )

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Self:
        # Extract the required columns
        inputs = []
        for input_item in df["inputs"].tolist():
            if isinstance(input_item, str):
                try:
                    # Try to parse as JSON dialogue
                    messages = json.loads(input_item)
                    if isinstance(messages, list):
                        dialogue = [Message(**msg) for msg in messages]
                        inputs.append(dialogue)
                    else:
                        # If not a list, treat as regular string input
                        inputs.append(input_item)
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as regular string input
                    inputs.append(input_item)

        labels = [Label(label) for label in df["labels"]]
        ids = [str(id) for id in df["ids"].tolist()]

        # Get all other columns as other_fields
        core_columns = {"inputs", "labels", "ids"}
        other_fields = {
            col: df[col].tolist() for col in df.columns if col not in core_columns
        }

        return cls(inputs=inputs, labels=labels, ids=ids, other_fields=other_fields)

    def to_records(self) -> Sequence[Record]:
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

    def to_pandas(self) -> pd.DataFrame:
        # Convert Dialogue inputs to dictionaries for pandas compatibility
        processed_inputs = []
        for input_item in self.inputs:
            if isinstance(input_item, str):
                processed_inputs.append(input_item)
            else:  # It's a Dialogue
                # Convert the entire dialogue to a single JSON string
                processed_inputs.append(
                    json.dumps([message.model_dump() for message in input_item])
                )

        base_data = {
            "inputs": processed_inputs,
            "labels": [label.value for label in self.labels],
            "ids": self.ids,
        }
        # Add each field from other_fields as a separate column
        base_data.update(self.other_fields)
        return pd.DataFrame(base_data)

    def labels_numpy(self) -> Float[np.ndarray, " batch_size"]:
        return np.array([label.to_int() for label in self.labels])

    @cached_property
    def stable_hash(self) -> str:
        return hashlib.sha256(self.to_pandas().to_csv().encode()).hexdigest()[:10]

    @overload
    def __getitem__(self, idx: int) -> Record: ...

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    def __getitem__(self, idx: int | slice | list[int]) -> Self | Record:
        if isinstance(idx, list):
            return type(self)(
                inputs=[self.inputs[i] for i in idx],
                labels=[self.labels[i] for i in idx],
                ids=[self.ids[i] for i in idx],
                other_fields={
                    k: [v[i] for i in idx] for k, v in self.other_fields.items()
                },
            )
        elif isinstance(idx, slice):
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
        return type(self).from_records([r for r in self.to_records() if filter_fn(r)])
