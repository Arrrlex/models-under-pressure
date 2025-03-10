import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from models_under_pressure.interfaces.dataset import (
    Dataset,
    Label,
    LabelledDataset,
    LabelledRecord,
    Message,
    Record,
)


@pytest.fixture
def sample_records() -> List[Record]:
    return [
        Record(
            input="test input 1", id="1", other_fields={"field1": "value1", "field2": 1}
        ),
        Record(
            input=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
            ],
            id="2",
            other_fields={"field1": "value2", "field2": 2},
        ),
    ]


@pytest.fixture
def sample_dataset(sample_records: List[Record]) -> Dataset:
    return Dataset.from_records(sample_records)


def test_dataset_creation(sample_dataset: Dataset) -> None:
    assert len(sample_dataset) == 2
    assert len(sample_dataset.inputs) == 2
    assert len(sample_dataset.ids) == 2
    assert all(k in sample_dataset.other_fields for k in ["field1", "field2"])


def test_dataset_indexing(sample_dataset: Dataset) -> None:
    # Test integer indexing
    record = sample_dataset[0]
    assert isinstance(record, Record)
    assert record.input == "test input 1"
    assert record.id == "1"

    # Test slice indexing
    subset = sample_dataset[0:1]
    assert isinstance(subset, Dataset)
    assert len(subset) == 1

    # Test list indexing
    subset = sample_dataset[[0, 1]]
    assert isinstance(subset, Dataset)
    assert len(subset) == 2


def test_dataset_sampling(sample_dataset: Dataset) -> None:
    sampled = sample_dataset.sample(1)
    assert isinstance(sampled, Dataset)
    assert len(sampled) == 1


def test_dataset_filtering(sample_dataset: Dataset) -> None:
    filtered = sample_dataset.filter(lambda r: r.id == "1")
    assert isinstance(filtered, Dataset)
    assert len(filtered) == 1
    assert filtered[0].id == "1"


def test_dataset_to_pandas(sample_dataset: Dataset, tmp_path: Path) -> None:
    df = sample_dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert all(col in df.columns for col in ["inputs", "ids", "field1", "field2"])


def test_dataset_save_load(sample_dataset: Dataset, tmp_path: Path) -> None:
    # Test CSV
    csv_path = tmp_path / "test.csv"
    sample_dataset.save_to(csv_path)
    loaded_csv = Dataset.load_from(csv_path)
    assert len(loaded_csv) == len(sample_dataset)

    # Test JSONL
    jsonl_path = tmp_path / "test.jsonl"
    sample_dataset.save_to(jsonl_path)
    loaded_jsonl = Dataset.load_from(jsonl_path)
    assert len(loaded_jsonl) == len(sample_dataset)


def test_dataset_from_pandas() -> None:
    df = pd.DataFrame(
        {
            "inputs": ["test1", json.dumps([{"role": "user", "content": "hello"}])],
            "ids": ["1", "2"],
            "extra": ["a", "b"],
        }
    )
    dataset = Dataset.from_pandas(df)
    assert len(dataset) == 2
    assert isinstance(dataset[0].input, str)
    assert isinstance(dataset[1].input, list)


def test_validation_errors(tmp_path: Path) -> None:
    # Test length mismatch
    with pytest.raises(ValueError):
        Dataset(inputs=["test1"], ids=["1", "2"], other_fields={"field1": ["a"]})

    # Test file exists error
    dataset = Dataset(inputs=["test1"], ids=["1"], other_fields={})
    with pytest.raises(FileExistsError):
        path = tmp_path / "test.csv"
        path.touch()
        dataset.save_to(path, overwrite=False)


def test_invalid_file_type(sample_dataset: Dataset, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sample_dataset.save_to(tmp_path / "test.txt")


def test_invalid_huggingface_dataset_name() -> None:
    with pytest.raises(ValueError):
        Dataset.load_from("invalid_dataset_name")


@pytest.fixture
def sample_labelled_records() -> List[LabelledRecord]:
    return [
        LabelledRecord(
            input="test input 1",
            id="1",
            other_fields={"labels": Label.LOW_STAKES.value, "field1": "value1"},
        ),
        LabelledRecord(
            input=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
            ],
            id="2",
            other_fields={"labels": Label.HIGH_STAKES.value, "field1": "value2"},
        ),
        LabelledRecord(
            input="test input 3",
            id="3",
            other_fields={"labels": Label.AMBIGUOUS.value, "field1": "value3"},
        ),
    ]


@pytest.fixture
def sample_labelled_dataset(
    sample_labelled_records: List[LabelledRecord],
) -> LabelledDataset:
    return LabelledDataset.from_records(sample_labelled_records)


def test_labelled_dataset_creation(sample_labelled_dataset: LabelledDataset) -> None:
    assert len(sample_labelled_dataset) == 3
    assert "labels" in sample_labelled_dataset.other_fields
    assert all(isinstance(label, Label) for label in sample_labelled_dataset.labels)


def test_labels_property(sample_labelled_dataset: LabelledDataset) -> None:
    labels = sample_labelled_dataset.labels
    assert len(labels) == 3
    assert labels[0] == Label.LOW_STAKES
    assert labels[1] == Label.HIGH_STAKES
    assert labels[2] == Label.AMBIGUOUS


def test_labels_numpy(sample_labelled_dataset: LabelledDataset) -> None:
    labels_array = sample_labelled_dataset.labels_numpy()
    assert isinstance(labels_array, np.ndarray)
    assert labels_array.shape == (3,)
    assert np.array_equal(labels_array, np.array([0, 1, 2]))


def test_labelled_dataset_indexing(sample_labelled_dataset: LabelledDataset) -> None:
    # Test integer indexing
    record = sample_labelled_dataset[0]
    assert isinstance(record, LabelledRecord)
    assert record.label == Label.LOW_STAKES

    # Test slice indexing
    subset = sample_labelled_dataset[1:3]
    assert isinstance(subset, LabelledDataset)
    assert len(subset) == 2
    assert subset.labels == [Label.HIGH_STAKES, Label.AMBIGUOUS]

    # Test list indexing
    subset = sample_labelled_dataset[[0, 2]]
    assert isinstance(subset, LabelledDataset)
    assert len(subset) == 2
    assert subset.labels == [Label.LOW_STAKES, Label.AMBIGUOUS]


def test_labelled_dataset_from_pandas() -> None:
    df = pd.DataFrame(
        {
            "inputs": ["test1", "test2"],
            "ids": ["1", "2"],
            "labels": [Label.LOW_STAKES.value, Label.HIGH_STAKES.value],
            "extra": ["a", "b"],
        }
    )
    dataset = LabelledDataset.from_pandas(df)
    assert len(dataset) == 2
    assert dataset.labels == [Label.LOW_STAKES, Label.HIGH_STAKES]


def test_labelled_dataset_validation_error() -> None:
    # Test missing labels field
    with pytest.raises(ValueError):
        LabelledDataset(
            inputs=["test1"],
            ids=["1"],
            other_fields={"field1": ["a"]},  # Missing labels field
        )


def test_labelled_dataset_filtering(sample_labelled_dataset: LabelledDataset) -> None:
    # Filter to keep only high stakes examples
    high_stakes = sample_labelled_dataset.filter(lambda r: r.label == Label.HIGH_STAKES)
    assert isinstance(high_stakes, LabelledDataset)
    assert len(high_stakes) == 1
    assert high_stakes.labels == [Label.HIGH_STAKES]


def test_labelled_record_label_property(
    sample_labelled_records: List[LabelledRecord],
) -> None:
    record = sample_labelled_records[0]
    assert record.label == Label.LOW_STAKES

    # Test with integer labels
    record_int = LabelledRecord(
        input="test",
        id="1",
        other_fields={"labels": 0},  # Using integer label
    )
    assert record_int.label == Label.LOW_STAKES


def test_label_conversion() -> None:
    # Test Label enum conversions
    assert Label.LOW_STAKES.to_int() == 0
    assert Label.HIGH_STAKES.to_int() == 1
    assert Label.AMBIGUOUS.to_int() == 2

    assert Label.from_int(0) == Label.LOW_STAKES
    assert Label.from_int(1) == Label.HIGH_STAKES
    assert Label.from_int(2) == Label.AMBIGUOUS

    with pytest.raises(KeyError):
        Label.from_int(3)  # Invalid label index


def test_label_value_consistency() -> None:
    """Test that Label enum values are consistent"""
    assert Label.LOW_STAKES.value == "low-stakes"
    assert Label.HIGH_STAKES.value == "high-stakes"
    assert Label.AMBIGUOUS.value == "ambiguous"


def test_label_bidirectional_conversion() -> None:
    """Test that converting to int and back preserves the label"""
    for label in Label:
        assert Label.from_int(label.to_int()) == label


def test_to_dialogue() -> None:
    """Test the to_dialogue conversion function"""
    from models_under_pressure.interfaces.dataset import to_dialogue

    # Test string input
    str_input = "hello world"
    dialogue = to_dialogue(str_input)
    assert len(dialogue) == 1
    assert dialogue[0].role == "user"
    assert dialogue[0].content == "hello world"

    # Test dialogue input
    dialogue_input = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]
    dialogue = to_dialogue(dialogue_input)
    assert len(dialogue) == 2
    assert dialogue == dialogue_input


def test_message_validation() -> None:
    """Test Message creation and validation"""
    # Valid messages
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"

    msg = Message(role="assistant", content="hi")
    assert msg.role == "assistant"
    assert msg.content == "hi"


def test_record_input_str() -> None:
    """Test Record.input_str() method"""
    # Test with string input
    record = Record(input="hello world", id="1")
    assert record.input_str() == "hello world"

    # Test with dialogue input
    record = Record(
        input=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ],
        id="2",
    )
    assert record.input_str() == "user: hello\nassistant: hi"
