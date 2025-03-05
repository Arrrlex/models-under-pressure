import hashlib
import json
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Self, Sequence, overload

import numpy as np
import pandas as pd
from jaxtyping import Float
from pydantic import BaseModel, Field, model_validator


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


class LabelledRecord(BaseModel):
    input: Input
    label: Label
    id: str
    other_fields: Mapping[str, Any]


class Record(BaseModel):
    input: Input
    id: str
    other_fields: Dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    inputs: Sequence[Input]
    ids: Sequence[str]
    other_fields: Mapping[str, Sequence[Any]]

    """
    Interface for a dataset class, the dataset is stored as a list of inputs, ids, and
    a mapping to 'other fields' which are arbitrary additional fields.

    The base dataset class is used to store the dataset in a way that is agnostic to the label field.
    """

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        """Verify that inputs, ids and each element of other_fields have the same length"""
        input_len = len(self.inputs)
        if len(self.ids) != input_len:
            raise ValueError(
                f"Length mismatch: inputs ({input_len}) != ids ({len(self.ids)})"
            )

        for field_name, field_values in self.other_fields.items():
            if len(field_values) != input_len:
                raise ValueError(
                    f"Length mismatch: inputs ({input_len}) != {field_name} ({len(field_values)})"
                )
        return self

    def __len__(self) -> int:
        return len(self.inputs)

    @overload
    def __getitem__(self, idx: int) -> Record: ...

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    def __getitem__(self, idx: int | slice | list[int]) -> Self | Record:
        if isinstance(idx, list):
            return type(self)(
                inputs=[self.inputs[i] for i in idx],
                ids=[self.ids[i] for i in idx],
                other_fields={
                    k: [v[i] for i in idx] for k, v in self.other_fields.items()
                },
            )
        elif isinstance(idx, slice):
            return type(self)(
                inputs=self.inputs[idx],
                ids=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
        else:
            return Record(
                input=self.inputs[idx],
                id=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )

    def to_records(self) -> Sequence[Record]:
        return [
            Record(
                input=input,
                id=id,
                other_fields={k: v[i] for k, v in self.other_fields.items()},
            )
            for i, (input, id) in enumerate(zip(self.inputs, self.ids))
        ]

    @classmethod
    def from_records(cls, records: Sequence[Record]) -> Self:
        field_keys = records[0].other_fields.keys()
        return cls(
            inputs=[r.input for r in records],
            ids=[r.id for r in records],
            other_fields={k: [r.other_fields[k] for r in records] for k in field_keys},
        )

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
            "ids": self.ids,
        }
        # Add each field from other_fields as a separate column
        base_data.update(self.other_fields)

        # Save base_data dictionary to file
        with open("temp_data/debug.json", "w") as f:
            json.dump(base_data, f)

        return pd.DataFrame(base_data)

    @cached_property
    def stable_hash(self) -> str:
        return hashlib.sha256(self.to_pandas().to_csv().encode()).hexdigest()[:10]

    def __len__(self) -> int:
        return len(self.inputs)

    @overload
    def __getitem__(self, idx: int) -> Record: ...

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    def __getitem__(self, idx: int | slice | list[int]) -> Self | Record:
        if isinstance(idx, list):
            return type(self)(
                inputs=[self.inputs[i] for i in idx],
                ids=[self.ids[i] for i in idx],
                other_fields={
                    k: [v[i] for i in idx] for k, v in self.other_fields.items()
                },
            )
        elif isinstance(idx, slice):
            return type(self)(
                inputs=self.inputs[idx],
                ids=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
        else:
            return Record(
                input=self.inputs[idx],
                id=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, field_mapping: Optional[Mapping[str, str]] = None
    ) -> Self:
        # Extract the required columns
        df = df.rename(columns=field_mapping or {})

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
            elif isinstance(input_item, list):
                dialogue = [Message(**msg) for msg in input_item]
                inputs.append(dialogue)
            else:
                raise ValueError(f"Invalid input type: {type(input_item)}")

        ids = [str(id) for id in df["ids"].tolist()]

        # Get all other columns as other_fields
        other_fields = {
            col: df[col].tolist() for col in df.columns if col not in {"inputs", "ids"}
        }

        return cls(inputs=inputs, ids=ids, other_fields=other_fields)

    @classmethod
    def from_jsonl(
        cls, file_path: Path, field_mapping: Optional[Mapping[str, str]] = None
    ) -> Self:
        with open(file_path, "r") as f:
            df = pd.DataFrame([json.loads(line) for line in f])

        return cls.from_pandas(df, field_mapping=field_mapping)

    @classmethod
    def from_csv(
        cls, file_path: Path, field_mapping: Optional[Mapping[str, str]] = None
    ) -> Self:
        df = pd.read_csv(file_path)
        return cls.from_pandas(df, field_mapping=field_mapping)

    @classmethod
    def load_from(
        cls,
        file_path: Path,
        field_mapping: Optional[Mapping[str, str]] = None,
        split: str = "train",
    ) -> "Dataset":
        """
        Load the dataset from a file, inferring type from extension if not specified.
        Supported types are:
        - csv: A CSV file with columns "input", "id", and other fields
        - jsonl: A JSONL file with each line being a JSON object with keys "input" and "id"
        - hf: A Hugging Face dataset, specified by a dataset name or path to a local file

        Args:
            file_path: The path to the file to load
            split: The split to load for HuggingFace datasets
        """
        # Infer from extension
        if str(file_path).endswith(".csv"):
            file_type = "csv"
        elif str(file_path).endswith(".jsonl"):
            file_type = "jsonl"
        else:
            # Assume HuggingFace dataset if no recognized extension
            file_type = "hf"

        if file_type == "csv":
            return cls.from_csv(file_path, field_mapping=field_mapping)

        elif file_type == "jsonl":
            return cls.from_jsonl(file_path, field_mapping=field_mapping)

        elif file_type == "hf":
            raise NotImplementedError("HF loading not implemented")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def filter(self, filter_fn: Callable[[Record], bool]) -> Self:
        return type(self).from_records([r for r in self.to_records() if filter_fn(r)])

    @cached_property
    def stable_hash(self) -> str:
        return hashlib.sha256(self.to_pandas().to_csv().encode()).hexdigest()[:10]


class LabelledDataset(Dataset):
    """
    A dataset with a "labels" field.
    """

    @model_validator(mode="after")
    def validate_label_name(self) -> Self:
        if self.other_fields.get("labels") is None:
            raise ValueError("labels column not found in other fields")
        return self

    @property
    def labels(self) -> Sequence[Label]:
        return [
            Label.from_int(label) if isinstance(label, int) else Label(label)
            for label in self.other_fields["labels"]
        ]

    def to_labelled_records(self) -> Sequence[LabelledRecord]:
        return [
            LabelledRecord(
                input=input,
                label=label,
                id=id,
                other_fields={k: v[i] for k, v in self.other_fields.items()},
            )
            for i, (input, label, id) in enumerate(
                zip(self.inputs, self.labels, self.ids)
            )
        ]

    def labels_numpy(self) -> Float[np.ndarray, " batch_size"]:
        return np.array([label.to_int() for label in self.labels])

    def filter(self, filter_fn: Callable[[LabelledRecord], bool]) -> Self:
        return type(self).from_records(
            [r for r in self.to_labelled_records() if filter_fn(r)]
        )


if __name__ == "__main__":
    from models_under_pressure.config import EVAL_DATASETS

    for key in EVAL_DATASETS:
        dataset_config = EVAL_DATASETS[key]
        dataset = LabelledDataset.load_from(
            file_path=dataset_config["path"],
            field_mapping=dataset_config["field_mapping"],
        )
        print(dataset[:5])
