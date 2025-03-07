import random

from datasets import load_dataset

from models_under_pressure.config import EVAL_DATASETS
from models_under_pressure.eval_datasets.label_dataset import label_dataset
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
    samples_per_class = num_samples // 2
    high_stakes = dataset.filter(lambda r: r.label == Label.HIGH_STAKES)
    low_stakes = dataset.filter(lambda r: r.label == Label.LOW_STAKES)

    if len(high_stakes) < samples_per_class:
        raise ValueError(
            f"Not enough high-stakes samples to create balanced dataset. "
            f"Need {samples_per_class} samples but only found {len(high_stakes)}."
        )

    if len(low_stakes) < samples_per_class:
        raise ValueError(
            f"Not enough low-stakes samples to create balanced dataset. "
            f"Need {samples_per_class} samples but only found {len(low_stakes)}."
        )

    high_stakes_sample = list(high_stakes.sample(samples_per_class).to_records())
    low_stakes_sample = list(low_stakes.sample(samples_per_class).to_records())

    balanced_records = high_stakes_sample + low_stakes_sample
    random.shuffle(balanced_records)

    return LabelledDataset.from_records(balanced_records)


def main(num_samples: int = 100, overwrite: bool = False):
    # Load data
    dataset = load_toolace_data(num_samples=num_samples * 5)

    # Label the data
    dataset = label_dataset(dataset)

    # Subsample the data
    print("Subsampling the data to get a balanced dataset")
    dataset = subsample_balanced_subset(dataset, num_samples=num_samples)

    # Save the data
    print(f"Saving the data to {EVAL_DATASETS['toolace']}")
    dataset.save_to(EVAL_DATASETS["toolace"], overwrite=overwrite)


if __name__ == "__main__":
    main(num_samples=100, overwrite=True)
