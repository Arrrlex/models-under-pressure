import random

from datasets import load_dataset

from models_under_pressure.config import EVAL_DATASETS_BALANCED, EVAL_DATASETS_RAW
from models_under_pressure.eval_datasets.label_dataset import (
    create_eval_dataset,
    create_test_dataset,
)
from models_under_pressure.interfaces.dataset import (
    Dataset,
    Message,
)


def load_toolace_raw_data(num_samples: int | None = None) -> Dataset:
    print("Loading ToolACE dataset from huggingface")
    ds = load_dataset("Team-ACE/ToolACE")["train"]  # type: ignore
    inputs = []
    all_ids = list(range(len(ds)))
    if num_samples is not None and len(ds) > num_samples:
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


def create_toolace_dataset(
    num_samples: int = 100, recompute: bool = False, split: str = "dev"
):
    dataset = load_toolace_raw_data(num_samples=num_samples)

    if split == "dev":
        return create_eval_dataset(
            dataset,
            raw_output_path=EVAL_DATASETS_RAW["toolace"],
            balanced_output_path=EVAL_DATASETS_BALANCED["toolace"],
            recompute=recompute,
        )
    elif split == "test":
        return create_test_dataset(
            dataset,
            dataset_name="toolace",
            recompute=recompute,
        )
    else:
        raise ValueError(f"Invalid split: {split}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute",
        type=bool,
        default=False,
        help="Recompute labels even if they already exist",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split to generate (dev or test)",
    )
    args = parser.parse_args()

    create_toolace_dataset(
        num_samples=args.num_samples,
        recompute=args.recompute,
        split=args.split,
    )
