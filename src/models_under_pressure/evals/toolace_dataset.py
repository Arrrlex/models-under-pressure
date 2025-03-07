import random

from datasets import load_dataset

from models_under_pressure.config import TOOLACE_SAMPLES_CSV
from models_under_pressure.evals.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import (
    Dataset,
    Label,
    LabelledDataset,
    Message,
)


def load_toolace_data(num_samples: int | None = None) -> Dataset:
    print("Loading ToolACE dataset from huggingface")
    ds = load_dataset("Team-ACE/ToolACE")["train"]  # type: ignore
    inputs = []
    all_ids = list(range(len(ds)))
    if num_samples is not None:
        ids = random.sample(all_ids, num_samples)
    else:
        ids = all_ids

    for ix in ids:
        row = ds[ix]
        system_prompt = row["system"]
        dialogue = [
            Message(role="system", content=system_prompt),
        ]
        for turn in row["conversations"]:
            dialogue.append(Message(role=turn["from"], content=turn["value"]))
        inputs.append(dialogue)

    return Dataset(inputs=inputs, ids=[str(i) for i in ids], other_fields={})


def subsample_balanced_subset(
    dataset: LabelledDataset, *, num_samples: int
) -> LabelledDataset:
    """Subsample a balanced subset of the dataset"""
    # Get the labels
    records = dataset.to_records()

    # Split records by label
    high_stakes = [r for r in records if r.label == Label.HIGH_STAKES]
    low_stakes = [r for r in records if r.label == Label.LOW_STAKES]

    # Check if we have enough high-stakes samples
    if len(high_stakes) < num_samples // 2:
        raise ValueError(
            f"Not enough high-stakes samples to create balanced dataset. "
            f"Need {num_samples // 2} samples but only found {len(high_stakes)}."
        )

    # Check if we have enough low-stakes samples
    if len(low_stakes) < num_samples // 2:
        raise ValueError(
            f"Not enough low-stakes samples to create balanced dataset. "
            f"Need {num_samples // 2} samples but only found {len(low_stakes)}."
        )

    # Sample equal numbers from each class
    samples_per_class = num_samples // 2
    high_stakes_sample = random.sample(high_stakes, samples_per_class)
    low_stakes_sample = random.sample(low_stakes, samples_per_class)

    # Combine samples
    balanced_records = high_stakes_sample + low_stakes_sample
    random.shuffle(balanced_records)

    return LabelledDataset.from_records(balanced_records)


def create_toolace_dataset(num_samples: int):
    # Load data
    data = load_toolace_data(num_samples=num_samples * 5 if num_samples else None)

    # Label the data
    dataset = label_dataset(data)

    dataset = subsample_balanced_subset(dataset, num_samples=num_samples)

    dataset.save_to(TOOLACE_SAMPLES_CSV)


if __name__ == "__main__":
    create_toolace_dataset(num_samples=100)
