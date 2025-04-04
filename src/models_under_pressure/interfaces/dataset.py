import json
import pickle
import random
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Literal,
    Mapping,
    Optional,
    Self,
    Sequence,
    Type,
    TypeVar,
    overload,
)

import datasets
import numpy as np
import pandas as pd
import torch
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


class Record(BaseModel):
    input: Input
    id: str
    other_fields: Dict[str, Any] = Field(default_factory=dict)

    def input_str(self) -> str:
        if isinstance(self.input, str):
            return self.input
        else:
            return "\n".join(
                f"{message.role}: {message.content}" for message in self.input
            )


class LabelledRecord(Record):
    @property
    def label(self) -> Label:
        label_ = self.other_fields["labels"]
        return Label.from_int(label_) if isinstance(label_, int) else Label(label_)


R = TypeVar("R", bound=Record)


class DatasetSpec(BaseModel):
    """
    A dataset spec is a specification for a dataset. It contains all information needed to load the dataset from a file.
    A dataset spec has:
      - a path (Path), which is the path to the dataset
      - indices (Sequence[int] or Literal["all"]), which are the indices of the records to load
      - field_mapping (Mapping[str, str]), which is a mapping of field names to the field names in the original dataset
      - loader_kwargs (Mapping[str, Any]), which are additional keyword arguments to pass to the loader
    """

    path: Path
    indices: Sequence[int] | Literal["all"] = "all"
    field_mapping: Mapping[str, str] = Field(default_factory=dict)
    loader_kwargs: Mapping[str, Any] = Field(default_factory=dict)


class BaseDataset(BaseModel, Generic[R]):
    """
    Interface for a dataset class, the dataset is stored as a list of inputs, ids, and
    a mapping to 'other fields' which are arbitrary additional fields.

    The base dataset class is used to store the dataset in a way that is agnostic to the label field.
    """

    inputs: Sequence[Input]
    ids: Sequence[str]
    other_fields: Mapping[str, Sequence[Any] | np.ndarray | torch.Tensor]
    _record_class: ClassVar[Type]

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
    def __getitem__(self, idx: int) -> R: ...

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    @overload
    def __getitem__(self, idx: list[int]) -> Self: ...

    def __getitem__(self, idx: int | slice | list[int]) -> Self | R:
        if isinstance(idx, list):
            indexed_other_fields = {}
            for key, value in self.other_fields.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    indexed_other_fields[key] = value[idx]
                else:
                    indexed_other_fields[key] = [v[i] for v in value for i in idx]
            return type(self)(
                inputs=[self.inputs[i] for i in idx],
                ids=[self.ids[i] for i in idx],
                other_fields=indexed_other_fields,
            )
        elif isinstance(idx, slice):
            return type(self)(
                inputs=self.inputs[idx],
                ids=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
        else:
            return self._record_class(
                input=self.inputs[idx],
                id=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )

    def sample(self, num_samples: int) -> Self:
        idxs = random.sample(range(len(self)), num_samples)
        return self[idxs]

    def filter(self, filter_fn: Callable[[R], bool]) -> Self:
        idxs = [i for i, r in enumerate(self.to_records()) if filter_fn(r)]
        return self[idxs]

    def assign(self, **kwargs: Sequence[Any] | np.ndarray | torch.Tensor) -> Self:
        """
        Assign new fields to the dataset (works like pandas assign)

        Args:
            kwargs: A mapping of field names to values. The values can be a sequence, a numpy array, or a torch tensor.

        Returns:
        """
        return type(self)(
            inputs=self.inputs,
            ids=self.ids,
            other_fields=dict(self.other_fields) | kwargs,
        )

    @classmethod
    def from_records(cls, records: Sequence[R]) -> Self:
        field_keys = records[0].other_fields.keys()
        return cls(
            inputs=[r.input for r in records],
            ids=[r.id for r in records],
            other_fields={k: [r.other_fields[k] for r in records] for k in field_keys},
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

        if "ids" in df.columns:
            ids = [str(id) for id in df["ids"].tolist()]
        else:
            ids = [str(i) for i in range(len(inputs))]

        # try removing values in case of error
        other_fields = {
            col: df[col].values.tolist()
            for col in df.columns
            if col not in {"inputs", "ids"}
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
    def from_huggingface(
        cls,
        dataset_name: str,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        field_mapping: Optional[Mapping[str, str]] = None,
    ) -> Self:
        ds = datasets.load_dataset(dataset_name, split=split, name=subset)
        df = pd.DataFrame(ds)  # type: ignore
        return cls.from_pandas(df, field_mapping=field_mapping)

    @classmethod
    def load_from(
        cls,
        file_path_or_name: Path | str,
        field_mapping: Optional[Mapping[str, str]] = None,
        **loader_kwargs: Any,
    ) -> Self:
        """
        Load the dataset from a file, inferring type from extension if not specified.
        Supported types are:
        - csv: A CSV file with columns "input", "id", and other fields
        - jsonl: A JSONL file with each line being a JSON object with keys "input" and "id"
        - hf: A Hugging Face dataset, specified by a dataset name or path to a local file

        Args:
            file_path: The path to the file to load
            loader_kwargs: Additional keyword arguments to pass to the loader
        """
        # Infer from extension
        if isinstance(file_path_or_name, Path):
            loaders = {
                ".csv": cls.from_csv,
                ".jsonl": cls.from_jsonl,
            }
            try:
                loader = loaders[file_path_or_name.suffix]
            except KeyError:
                raise ValueError(f"Unsupported file type: '{file_path_or_name.suffix}'")
            return loader(
                file_path_or_name, field_mapping=field_mapping, **loader_kwargs
            )
        else:
            if not len(file_path_or_name.split("/")) == 2:
                raise ValueError(f"Invalid dataset name: {file_path_or_name}")
            return cls.from_huggingface(
                file_path_or_name, field_mapping=field_mapping, **loader_kwargs
            )

    @classmethod
    def concatenate(cls, datasets: Sequence[Self]) -> Self:
        return cls.from_records(
            [r for dataset in datasets for r in dataset.to_records()]
        )

    def to_records(self) -> Sequence[R]:
        return [
            self._record_class(
                input=input,
                id=id,
                other_fields={k: v[i] for k, v in self.other_fields.items()},
            )
            for i, (input, id) in enumerate(zip(self.inputs, self.ids))
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
            "ids": self.ids,
        }
        # Add each field from other_fields as a separate column
        processed_fields = {}
        for field_name, field_values in self.other_fields.items():
            processed_values = []
            for value in field_values:
                # Convert Label enum to string if needed
                if isinstance(value, Label):
                    processed_values.append(value.value)
                else:
                    processed_values.append(value)
            processed_fields[field_name] = processed_values

        # Add processed fields to base_data
        base_data.update(processed_fields)

        try:
            df = pd.DataFrame(base_data)
        except ValueError:
            # Store base_data as a pickle file to not lose any data
            print("Failed to convert to pandas, storing as pickle")
            with open("temp_base_data.pkl", "wb") as f:
                pickle.dump(base_data, f)

            print("Attempting to fix by removing unaligned columns ...")
            base_data = {
                k: v for k, v in base_data.items() if len(v) == len(base_data["inputs"])
            }
            df = pd.DataFrame(base_data)
        return df

    def save_to(self, file_path: Path, overwrite: bool = False) -> None:
        if not overwrite and file_path.exists():
            raise FileExistsError(
                f"File {file_path} already exists. Use overwrite=True to overwrite."
            )
        if file_path.suffix == ".csv":
            self.to_pandas().to_csv(file_path, index=False)
        elif file_path.suffix == ".jsonl":
            self.to_pandas().to_json(file_path, orient="records", lines=True)
        elif file_path.suffix == ".json":
            self.to_pandas().to_json(file_path, orient="records")
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")


class Dataset(BaseDataset[Record]):
    _record_class: ClassVar[Type] = Record


class LabelledDataset(BaseDataset[LabelledRecord]):
    """
    A dataset with a "labels" field.
    """

    _record_class: ClassVar[Type] = LabelledRecord

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

    def labels_numpy(self) -> Float[np.ndarray, " batch_size"]:
        return np.array([label.to_int() for label in self.labels])

    def print_label_distribution(self) -> Dict[str, float]:
        """
        Calculates and prints the distribution of labels in the dataset.

        Returns:
            A dictionary mapping label names to their percentage in the dataset
        """
        if len(self) == 0:
            print("Dataset is empty")
            return {}

        # Count occurrences of each label
        label_counts = {}
        for label in self.labels:
            label_name = label.value
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

        # Calculate percentages
        total = len(self)
        label_percentages = {
            label: (count / total) * 100 for label, count in label_counts.items()
        }

        # Print the distribution
        print(f"Label distribution (total: {total} examples):")
        for label, percentage in sorted(label_percentages.items()):
            count = label_counts[label]
            print(f"  {label}: {count} examples ({percentage:.2f}%)")

        return label_percentages


def subsample_balanced_subset(dataset: LabelledDataset) -> LabelledDataset:
    """Subsample a balanced subset of the dataset"""
    high_stakes = dataset.filter(lambda r: r.label == Label.HIGH_STAKES)
    low_stakes = dataset.filter(lambda r: r.label == Label.LOW_STAKES)

    if len(high_stakes) > len(low_stakes):
        high_stakes_sample = list(high_stakes.sample(len(low_stakes)).to_records())
        low_stakes_sample = list(low_stakes.to_records())
    else:
        high_stakes_sample = list(high_stakes.to_records())
        low_stakes_sample = list(low_stakes.sample(len(high_stakes)).to_records())

    balanced_records = high_stakes_sample + low_stakes_sample
    random.shuffle(balanced_records)

    return LabelledDataset.from_records(balanced_records)


if __name__ == "__main__":
    from models_under_pressure.config import EVAL_DATASETS

    for name, path in EVAL_DATASETS.items():
        dataset = LabelledDataset.load_from(path)
        print(dataset[:5])
