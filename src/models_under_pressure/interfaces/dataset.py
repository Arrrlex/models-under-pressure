import hashlib
import json
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Dict, Literal, Mapping, Self, Sequence, overload

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
    def from_pandas(cls, df: pd.DataFrame) -> "Dataset":
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

        ids = [str(id) for id in df["ids"].tolist()]

        # Get all other columns as other_fields
        core_columns = {"inputs", "labels", "ids"}
        other_fields = {
            col: df[col].tolist() for col in df.columns if col not in core_columns
        }

<<<<<<< HEAD
=======
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

        # Save base_data dictionary to file
        with open("temp_data/debug.json", "w") as f:
            json.dump(base_data, f)

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


class UnlabeledDataset(BaseModel):
    inputs: Sequence[Input]
    ids: Sequence[str]
    other_fields: Mapping[str, Sequence[Any]]

    def records(self) -> Sequence[UnlabeledRecord]:
        return [
            UnlabeledRecord(
                input=input,
                id=id,
                other_fields={k: v[i] for k, v in self.other_fields.items()},
            )
            for i, (input, id) in enumerate(zip(self.inputs, self.ids))
        ]

    def __len__(self) -> int:
        return len(self.inputs)

    @classmethod
    def from_records(cls, records: Sequence[UnlabeledRecord]) -> Self:
        field_keys = records[0].other_fields.keys()
        return cls(
            inputs=[r.input for r in records],
            ids=[r.id for r in records],
            other_fields={k: [r.other_fields[k] for r in records] for k in field_keys},
        )

    def to_pandas(self) -> pd.DataFrame:
        print("Converting dataset to pandas DataFrame")
        print(f"Number of inputs: {len(self.inputs)}")
        print(f"Number of ids: {len(self.ids)}")

        processed_inputs = []
        for input_item in self.inputs:
            if isinstance(input_item, str):
                processed_inputs.append(input_item)
            else:  # It's a Dialogue
                processed_inputs.append(
                    json.dumps([message.model_dump() for message in input_item])
                )

        base_data = {
            "inputs": processed_inputs,
            "ids": self.ids,
        }
        base_data.update(self.other_fields)

        print(f"Processed data shape: {len(processed_inputs)} rows")
        return pd.DataFrame(base_data)

    @overload
    def __getitem__(self, idx: int) -> UnlabeledRecord: ...

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    def __getitem__(self, idx: int | slice) -> Self | UnlabeledRecord:
        if isinstance(idx, slice):
            return type(self)(
                inputs=self.inputs[idx],
                ids=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
        else:
            return UnlabeledRecord(
                input=self.inputs[idx],
                id=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "UnlabeledDataset":
        print(f"Loading dataset from pandas DataFrame with shape: {df.shape}")
        inputs = []
        for input_item in df["inputs"].tolist():
            if isinstance(input_item, str):
                try:
                    messages = json.loads(input_item)
                    if isinstance(messages, list):
                        dialogue = [Message(**msg) for msg in messages]
                        inputs.append(dialogue)
                    else:
                        inputs.append(input_item)
                except json.JSONDecodeError:
                    print(
                        f"Warning: JSON decode error for input: {input_item[:100]}..."
                    )
                    inputs.append(input_item)
            else:
                print(input_item)
                raise NotImplementedError("Non-string inputs not supported")

        ids = [str(id) for id in df["ids"].tolist()]
        print(f"Processed {len(inputs)} inputs and {len(ids)} ids")

        core_columns = {"inputs", "ids"}
        other_fields = {
            col: df[col].tolist() for col in df.columns if col not in core_columns
        }
        print(f"Found {len(other_fields)} other fields")

>>>>>>> 0433f57 (fixed bug with filter calling records instead of to_records)
        return cls(inputs=inputs, ids=ids, other_fields=other_fields)

    @classmethod
    def from_jsonl(
        cls, file_path: str, input_name: str, id_name: str | None = None
    ) -> "Dataset":
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]

        inputs = [d[input_name] for d in data]
        ids = [
            d[id_name] if id_name is not None else str(i) for i, d in enumerate(data)
        ]

        other_field_keys = set(data[0].keys()) - {"prompt", "high_stakes", "id"}
        other_fields = {k: [d[k] for d in data] for k in other_field_keys}

        return Dataset(
            inputs=inputs,
            ids=ids,
            other_fields=other_fields,
        )

    @classmethod
    def load_from(
        cls,
        file_path: str,
        file_type: Literal["csv", "jsonl", "hf"],
        input_name: str,
        ids_name: str | None = None,
        split: str = "train",
    ) -> "Dataset":
        """
        Load the dataset from a file of a given type, supported types are:
        - csv: A CSV file with columns "input", "id", and other fields
        - jsonl: A JSONL file with each line being a JSON object with keys "input" and "id"
        - hf: A Hugging Face dataset, specified by a dataset name or path to a local file

        Args:
            file_path: The path to the file to load
            file_type: The type of the file to load
            input_name: The name of the column in the file that contains the input
            ids_name: The name of the column in the file that contains the ids
        """

        if file_type == "csv":
            df = pd.read_csv(file_path)

            # Change input_name to inputs:
            df = df.rename(columns={input_name: "inputs"})

            # If the input column is in a messages format i.e. {'role': 'user', 'content': '...'}
            # then we need to convert it to a dialogue
            try:
                df["inputs"] = df["inputs"].apply(json.loads)
                df["inputs"] = df["inputs"].apply(to_dialogue)

            except json.JSONDecodeError:
                print(
                    f"Failed to convert {input_name} to dialogue - invalid JSON format"
                )

            # Change ids_name to ids or generate ids if ids_name is None
            if ids_name is not None:
                df = df.rename(columns={ids_name: "ids"})
            else:
                df["ids"] = list(range(len(df)))

            return cls.from_pandas(df)

        elif file_type == "jsonl":
            return cls.from_jsonl(
                file_path=file_path, input_name=input_name, id_name=ids_name
            )

        elif file_type == "hf":
            raise NotImplementedError("HF loading not implemented")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def filter(self, filter_fn: Callable[[Record], bool]) -> Self:
        return type(self).from_records([r for r in self.to_records() if filter_fn(r)])


class LabelledDataset(Dataset):
    label_name: str

    """
    A dataset with a label field, the label field is stored in the other_fields dictionary
    under the key label_name but accessed as a property called labels.

    The class has specific loading methods for datasets or inputs specific to the class.

    The class reimplements the __getitem__ and to_<type> methods to include the label field.
    """

    @model_validator(mode="after")
    def validate_label_name(self) -> Self:
        if self.other_fields.get(self.label_name) is None:
            raise ValueError(f"Label name {self.label_name} not found in other fields")
        return self

    @property
    def labels(self) -> Sequence[Label]:
        return [Label(label) for label in self.other_fields[self.label_name]]

    def to_records(self) -> Sequence[LabelledRecord]:
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

    @overload
    def __getitem__(self, idx: int) -> LabelledRecord: ...

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    def __getitem__(self, idx: int | slice) -> Self | LabelledRecord:
        if isinstance(idx, slice):
            return type(self)(
                inputs=self.inputs[idx],
                ids=self.ids[idx],
                label_name=self.label_name,
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
        else:
            return LabelledRecord(
                input=self.inputs[idx],
                label=self.labels[idx],
                id=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
