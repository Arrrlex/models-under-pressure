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

import numpy as np
import pandas as pd
from jaxtyping import Float
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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
    """
    A record is a single example from a dataset.

    A record has:
      - an input (str or Dialogue)
      - an id (str), which should come from the original dataset
      - an index (int), which is the index of the record in the dataset
      - other fields (Dict[str, Any]), which are arbitrary additional fields

    Additionally, a record has:
     - a path (Path), which is the path to the original dataset
     - a field_mapping (Mapping[str, str]), which is a mapping of field names to the field names in the original dataset
     - a loader_kwargs (Mapping[str, Any]), which are additional keyword arguments to pass to the loader

     path, field_mapping, and loader_kwargs ensure that the Dataset object knows how to
     load the dataset from a file.
    """

    path: Path
    field_mapping: Mapping[str, str]
    loader_kwargs: Mapping[str, Any]
    input: Input
    id: str
    index: int
    other_fields: Dict[str, Any] = Field(default_factory=dict)

    def input_str(self) -> str:
        if isinstance(self.input, str):
            return self.input
        else:
            return "\n".join(
                f"{message.role}: {message.content}" for message in self.input
            )


class LabelledRecord(Record):
    """
    A labelled record is a record with a label field.
    """

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
    Generic base class for all datasets.

    A base dataset has:
      - a path (Path), which is the path to the dataset
      - indices (Sequence[int] or Literal["all"]), which are the indices of the records to load
      - field_mapping (Mapping[str, str]), which is a mapping of field names to the field names in the original dataset
      - loader_kwargs (Mapping[str, Any]), which are additional keyword arguments to pass to the loader
      - inputs (Sequence[Input]), which are the inputs of the dataset
      - ids (Sequence[str]), which are the ids of the dataset
      - other_fields (Mapping[str, Sequence[Any]]), which are arbitrary additional fields
      - _record_class (ClassVar[Type]), which is the class of the records in the dataset

    BaseDatasets store data in columnar format with each column being a sequence of values.

    BaseDatasets contain path, indices, field_mapping, and loader_kwargs, so that
    the Dataset can be loaded from a file. This information is preserved when we
    modify the dataset by filtering, indexing or subsampling. This is useful e.g.
    for retrieving stored activations.
    """

    path: Path
    indices: Sequence[int]
    field_mapping: Mapping[str, str] = Field(default_factory=dict)
    loader_kwargs: Mapping[str, Any] = Field(default_factory=dict)
    inputs: Sequence[Input]
    ids: Sequence[str]
    other_fields: Mapping[str, Sequence[Any]]
    _record_class: ClassVar[Type]

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

        if self.indices != "all":
            if len(self.indices) != input_len:
                raise ValueError(
                    f"Length mismatch: inputs ({input_len}) != indices ({len(self.indices)})"
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
    def __getitem__(self, idx: int) -> R:
        """
        Get the record at index `idx`.

        Note: when using square bracket indexing (e.g. dataset[0]), the index refers to the
        position in the current dataset's sequence, not the original dataset indices stored
        in self.indices. For example, if you have a dataset with 1000 records and filter it
        to keep only records 100-199, accessing filtered_dataset[0] will give you the first
        record in the filtered dataset (which was record 100 in the original dataset). The
        original indices are preserved in self.indices but are not used for indexing. For example:

            full_dataset = LabelledDataset.load_from(path)  # 1000 records
            filtered = full_dataset[100:200]  # Keeps records 100-199
            record = filtered[0]  # Gets record 100 from original dataset
        """

    @overload
    def __getitem__(self, idx: slice) -> Self:
        """
        Get a slice of the dataset.

        Note: when using square bracket indexing (e.g. dataset[0]), the index refers to the
        position in the current dataset's sequence, not the original dataset indices stored
        in self.indices. For example, if you have a dataset with 1000 records and filter it
        to keep only records 100-199, accessing filtered_dataset[0] will give you the first
        record in the filtered dataset (which was record 100 in the original dataset). The
        original indices are preserved in self.indices but are not used for indexing. For example:

            full_dataset = LabelledDataset.load_from(path)  # 1000 records
            filtered = full_dataset[100:200]  # Keeps records 100-199
            record = filtered[0]  # Gets record 100 from original dataset

        """

    @overload
    def __getitem__(self, idx: list[int]) -> Self:
        """
        Get a list of records at indices `idx`.

        Note: when using square bracket indexing (e.g. dataset[0]), the index refers to the
        position in the current dataset's sequence, not the original dataset indices stored
        in self.indices. For example, if you have a dataset with 1000 records and filter it
        to keep only records 100-199, accessing filtered_dataset[0] will give you the first
        record in the filtered dataset (which was record 100 in the original dataset). The
        original indices are preserved in self.indices but are not used for indexing. For example:

            full_dataset = LabelledDataset.load_from(path)  # 1000 records
            filtered = full_dataset[100:200]  # Keeps records 100-199
            record = filtered[0]  # Gets record 100 from original dataset
        """

    def __getitem__(self, idx: int | slice | list[int]) -> Self | R:
        if isinstance(idx, list):
            return type(self)(
                path=self.path,
                field_mapping=self.field_mapping,
                loader_kwargs=self.loader_kwargs,
                indices=[self.indices[i] for i in idx],
                inputs=[self.inputs[i] for i in idx],
                ids=[self.ids[i] for i in idx],
                other_fields={
                    k: [v[i] for i in idx] for k, v in self.other_fields.items()
                },
            )
        elif isinstance(idx, slice):
            return type(self)(
                path=self.path,
                field_mapping=self.field_mapping,
                loader_kwargs=self.loader_kwargs,
                indices=self.indices[idx],
                inputs=self.inputs[idx],
                ids=self.ids[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
            )
        else:
            return self._record_class(
                input=self.inputs[idx],
                id=self.ids[idx],
                index=self.indices[idx],
                other_fields={k: v[idx] for k, v in self.other_fields.items()},
                path=self.path,
                field_mapping=self.field_mapping,
                loader_kwargs=self.loader_kwargs,
            )

    def sample(self, num_samples: int) -> Self:
        """
        Sample a random subset of `num_samples` records from the dataset.
        """
        return type(self).from_records(random.sample(self.to_records(), num_samples))

    def filter(self, filter_fn: Callable[[R], bool]) -> Self:
        """
        Filter the dataset by a boolean function `filter_fn` that returns True for
        records to keep and False for records to discard.
        """
        return type(self).from_records([r for r in self.to_records() if filter_fn(r)])

    @property
    def spec(self) -> DatasetSpec:
        """
        Get the dataset spec for the dataset.
        """
        indices = "all" if self.indices == list(range(len(self))) else self.indices
        return DatasetSpec(
            path=self.path,
            indices=indices,
            field_mapping=self.field_mapping,
            loader_kwargs=self.loader_kwargs,
        )

    @classmethod
    def from_records(cls, records: Sequence[R]) -> Self:
        """
        Create a new dataset from a sequence of records.

        Args:
            records: A sequence of records
        """
        field_keys = records[0].other_fields.keys()
        return cls(
            path=records[0].path,
            field_mapping=records[0].field_mapping,
            loader_kwargs=records[0].loader_kwargs,
            indices=[r.index for r in records],
            inputs=[r.input for r in records],
            ids=[r.id for r in records],
            other_fields={k: [r.other_fields[k] for r in records] for k in field_keys},
        )

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        path: Path,
        field_mapping: Optional[Mapping[str, str]] = None,
    ) -> Self:
        """
        Create a new dataset from a pandas DataFrame.

        Args:
            df: A pandas DataFrame
            path: The path to the dataset
            field_mapping: A mapping of field names to the field names in the original dataset

        Though the data is already present in `df`, we need `path` and `field_mapping`
        to know how to load the dataset from a file, an important design consideration
        for Dataset objects.

        This method handles parsing the inputs and ids from the DataFrame, and
        converting them to the correct format for the Dataset object.
        """
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

        return cls(
            path=path,
            indices=list(range(len(inputs))),
            field_mapping=field_mapping or {},
            loader_kwargs={},
            inputs=inputs,
            ids=ids,
            other_fields=other_fields,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        field_mapping: Optional[Mapping[str, str]] = None,
    ) -> Self:
        """
        Create a new dataset from a JSONL file.

        Args:
            path: The path to the JSONL file
            field_mapping: A mapping of field names to the field names in the original dataset
        """
        with open(path, "r") as f:
            df = pd.DataFrame([json.loads(line) for line in f])

        return cls.from_pandas(df, path=path, field_mapping=field_mapping)

    @classmethod
    def from_csv(
        cls,
        path: Path,
        field_mapping: Optional[Mapping[str, str]] = None,
    ) -> Self:
        """
        Create a new dataset from a CSV file.

        Args:
            path: The path to the CSV file
            field_mapping: A mapping of field names to the field names in the original dataset
        """
        df = pd.read_csv(path)
        return cls.from_pandas(df, path=path, field_mapping=field_mapping)

    @classmethod
    def load_from(
        cls,
        path: Path,
        field_mapping: Optional[Mapping[str, str]] = None,
        indices: Sequence[int] | Literal["all"] = "all",
        **loader_kwargs: Any,
    ) -> Self:
        """
        Load the dataset from a file, inferring type from extension if not specified.
        Supported types are:
        - csv: A CSV file with columns "input", "id", and other fields
        - jsonl: A JSONL file with each line being a JSON object with keys "input" and "id"

        Args:
            path: Path to the file to load
            field_mapping: Optional mapping of field names
            indices: Which indices to load, or "all" for all indices
            loader_kwargs: Additional keyword arguments to pass to the loader
        """
        spec = DatasetSpec(
            path=path,
            indices=indices,
            field_mapping=field_mapping or {},
            loader_kwargs=loader_kwargs,
        )
        return cls.load_from_spec(spec)

    @classmethod
    def load_from_spec(cls, spec: DatasetSpec) -> Self:
        """
        Load the dataset from a dataset spec.

        Args:
            spec: The dataset spec
        """
        # Infer from extension
        loaders = {
            ".csv": cls.from_csv,
            ".jsonl": cls.from_jsonl,
        }
        try:
            loader = loaders[spec.path.suffix]
        except KeyError:
            raise ValueError(f"Unsupported file type: '{spec.path.suffix}'")
        dataset_all_indices = loader(
            PROJECT_ROOT / spec.path,
            field_mapping=spec.field_mapping,
            **spec.loader_kwargs,
        )

        if spec.indices == "all":
            return dataset_all_indices
        else:
            return dataset_all_indices[list(spec.indices)]

    @classmethod
    def concatenate(cls, datasets: Sequence[Self]) -> Self:
        """
        Concatenate a sequence of datasets.

        Args:
            datasets: A sequence of datasets
        """
        if not datasets:
            raise ValueError("Cannot concatenate empty sequence of datasets")

        # Check that all datasets have consistent metadata
        first = datasets[0]
        for dataset in datasets[1:]:
            if dataset.path != first.path:
                raise ValueError("All datasets must have the same path")
            if dataset.field_mapping != first.field_mapping:
                raise ValueError("All datasets must have the same field mapping")
            if dataset.loader_kwargs != first.loader_kwargs:
                raise ValueError("All datasets must have the same loader kwargs")
        return cls.from_records(
            [r for dataset in datasets for r in dataset.to_records()]
        )

    def to_records(self) -> Sequence[R]:
        """
        Convert the dataset to a sequence of records.
        """
        return [
            self._record_class(
                input=input,
                path=self.path,
                field_mapping=self.field_mapping,
                loader_kwargs=self.loader_kwargs,
                id=id,
                index=self.indices[i],
                other_fields={k: v[i] for k, v in self.other_fields.items()},
            )
            for i, (input, id) in enumerate(zip(self.inputs, self.ids))
        ]

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.
        """
        # Convert Dialogue inputs to JSON strings for pandas compatibility
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
        """
        Save the dataset to a file.

        Args:
            file_path: The path to save the dataset to
            overwrite: Whether to overwrite the file if it already exists
        """
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
    """
    A dataset with no label field.
    """

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
