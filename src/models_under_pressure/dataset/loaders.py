import json
from pathlib import Path
from typing import Literal

import pandas as pd

from models_under_pressure.config import ANTHROPIC_SAMPLES_CSV, TOOLACE_SAMPLES_CSV
from models_under_pressure.interfaces.dataset import Dataset, Label


def load_anthropic_csv(filename: Path = ANTHROPIC_SAMPLES_CSV) -> Dataset:
    df = pd.read_csv(filename)

    messages = df["messages"].apply(json.loads)

    # Convert high_stakes column to Label enum
    labels = df["high_stakes"].apply(Label.from_int)

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


def load_toolace_csv(filename: Path = TOOLACE_SAMPLES_CSV) -> Dataset:
    dataset = Dataset.from_pandas(pd.read_csv(filename))
    return dataset


def load_jsonl(
    filename: Path,
    field_mapping: dict[str, str],
    label_format: Literal["int", "str"] = "int",
) -> Dataset:
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]

    inputs_key = field_mapping["prompt"]
    labels_key = field_mapping["label"]
    id_key = field_mapping["id"]
    other_keys = set(data[0].keys()) - {inputs_key, labels_key, id_key}

    inputs = [d[inputs_key] for d in data]
    convert_label = Label.from_int if label_format == "int" else Label
    labels = [convert_label(d[labels_key]) for d in data]
    ids = [d[id_key] for d in data]

    other_fields = {k: [d[k] for d in data] for k in other_keys}

    return Dataset(
        inputs=inputs,
        labels=labels,
        ids=ids,
        other_fields=other_fields,
    )


def load_generated_jsonl(filename: Path) -> Dataset:
    return load_jsonl(
        filename,
        field_mapping={"prompt": "prompt", "label": "high_stakes", "id": "id"},
        label_format="int",
    )
