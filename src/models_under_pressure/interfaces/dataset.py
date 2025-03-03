import json
import os
from enum import Enum
from typing import Any, Callable, Dict, Literal, Mapping, Self, Sequence, overload

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset, load_from_disk
from jaxtyping import Float
from pydantic import BaseModel, Field


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


class UnlabeledRecord(BaseModel):
    input: Input
    id: str
    other_fields: Dict[str, Any] = Field(default_factory=dict)


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

        labels = [Label(label) for label in df["labels"]]
        ids = [str(id) for id in df["ids"].tolist()]

        # Get all other columns as other_fields
        core_columns = {"inputs", "labels", "ids"}
        other_fields = {
            col: df[col].tolist() for col in df.columns if col not in core_columns
        }

        return cls(inputs=inputs, labels=labels, ids=ids, other_fields=other_fields)

    def filter(self, filter_fn: Callable[[Record], bool]) -> Self:
        return type(self).from_records([r for r in self.records() if filter_fn(r)])


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

        return cls(inputs=inputs, ids=ids, other_fields=other_fields)

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        input_name: str,
        ids_name: str | None = None,
    ) -> "UnlabeledDataset":
        df = pd.read_csv(file_path)

        # Change the input column name to "inputs"
        df.rename(columns={input_name: "inputs"}, inplace=True)

        # Change the ids column name to "id"
        if ids_name is not None:
            df.rename(columns={ids_name: "ids"}, inplace=True)
        else:
            # Set the ids to be a list of value from 0 to len(df)
            df["ids"] = list(range(len(df)))

        return cls.from_pandas(df)

    @classmethod
    def from_hf(
        cls,
        file_path: str,
        input_name: str,
        ids_name: str | None = None,
        split: str = "train",
    ) -> "UnlabeledDataset":
        """Load a dataset from a Hugging Face dataset.

        Args:
            file_path: Path to local dataset or name of HF dataset
            input_name: Name of input column in dataset
            ids_name: Optional name of ID column
            split: Dataset split to use, defaults to 'train'
        """
        print(f"file_path exists: {os.path.exists(file_path)}")

        if os.path.exists(file_path):
            hf_dataset = load_from_disk(file_path)
        else:
            hf_dataset = load_dataset(file_path)

        # Deal with the different possible types of Dataset HuggingFace can return
        assert isinstance(hf_dataset, HFDataset) or isinstance(
            hf_dataset, DatasetDict
        ), "Iterable Dataset or DatasetDict not supported"

        if isinstance(hf_dataset, DatasetDict):
            assert split is not None, "Split is required for DatasetDict"
            hf_dataset = hf_dataset[split]

        # TODO: rename columns
        # Rename columns and handle missing IDs
        renamed_dataset = {}
        renamed_dataset["input"] = hf_dataset[input_name]

        if ids_name is not None:
            renamed_dataset["id"] = hf_dataset[ids_name]
        else:
            renamed_dataset["id"] = list(range(len(hf_dataset)))

        # Copy over any other columns
        for col in hf_dataset.column_names:
            if col not in {input_name, ids_name}:
                renamed_dataset[col] = hf_dataset[col]

        hf_dataset = HFDataset.from_dict(renamed_dataset)

        df = hf_dataset.to_pandas()
        if isinstance(df, pd.DataFrame):
            return cls.from_pandas(df)
        else:
            # If it's an iterator, take the first DataFrame
            raise NotImplementedError("Iterator of DataFrames not supported")

    def filter(self, filter_fn: Callable[[UnlabeledRecord], bool]) -> Self:
        return type(self).from_records([r for r in self.records() if filter_fn(r)])

    @classmethod
    def load_from(
        cls,
        file_path: str,
        file_type: Literal["csv", "jsonl", "hf"],
        input_name: str,
        ids_name: str | None = None,
        split: str = "train",
    ) -> "UnlabeledDataset":
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
            return cls.from_csv(file_path, input_name, ids_name)

        elif file_type == "jsonl":
            raise NotImplementedError("JSONL loading not implemented")

        elif file_type == "hf":
            return cls.from_hf(file_path, input_name, ids_name, split)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")
