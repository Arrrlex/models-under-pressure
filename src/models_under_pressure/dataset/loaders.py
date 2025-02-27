import json
from pathlib import Path

import pandas as pd

from models_under_pressure.config import ANTHROPIC_SAMPLES_CSV
from models_under_pressure.interfaces.dataset import Dataset, Label


def load_anthropic_csv(filename: Path = ANTHROPIC_SAMPLES_CSV) -> Dataset:
    df = pd.read_csv(filename)

    messages = df["messages"].apply(json.loads)

    # Convert high_stakes column to Label enum
    labels = df["high_stakes"].apply(
        lambda x: Label.HIGH_STAKES
        if x == 1
        else Label.LOW_STAKES
        if x == 0
        else Label.AMBIGUOUS  # type: ignore
    )

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
