import random

from datasets import load_dataset

from models_under_pressure.config import EVAL_DATASETS, TOOLACE_SAMPLES_CSV
from models_under_pressure.evals.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import Dataset, LabelledDataset, Message


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


def create_toolace_dataset(num_samples: int | None = None):
    # Load data
    data = load_toolace_data(num_samples=num_samples)

    # Label the data
    dataset = label_dataset(data)

    dataset.to_pandas().to_csv(TOOLACE_SAMPLES_CSV, index=False)


if __name__ == "__main__":
    create_toolace_dataset(num_samples=500)

    dataset = LabelledDataset.load_from(**EVAL_DATASETS["toolace"])
    print(dataset[0])
