import random

from datasets import load_dataset

from models_under_pressure.config import EVAL_DATASETS_BALANCED, EVAL_DATASETS_RAW
from models_under_pressure.eval_datasets.label_dataset import create_eval_dataset
from models_under_pressure.interfaces.dataset import (
    Dataset,
    Message,
)


def load_toolace_raw_data(num_samples: int | None = None) -> Dataset:
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


def create_toolace_dataset(num_samples: int = 100, recompute: bool = False):
    dataset = load_toolace_raw_data(num_samples=num_samples)
    return create_eval_dataset(
        dataset,
        raw_output_path=EVAL_DATASETS_RAW["toolace"],
        balanced_output_path=EVAL_DATASETS_BALANCED["toolace"],
        recompute=recompute,
    )


if __name__ == "__main__":
    create_toolace_dataset(num_samples=100)
