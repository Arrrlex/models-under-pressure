import random
from datasets import load_dataset
import numpy as np
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import (
    Dataset,
    Record,
    Message,
    subsample_balanced_subset,
)
from models_under_pressure.utils import hf_login
from models_under_pressure.config import EVALS_DIR, TEST_EVALS_DIR

hf_login()


def load_mask_dataset():
    records = []
    dataset_subsets = {
        subset: load_dataset("cais/MASK", name=subset, split="test")
        for subset in ["continuations", "known_facts", "statistics"]
    }
    # Calculate intersection of keys across all subsets
    common_keys = set().intersection(
        *[set(ds[0].keys()) for ds in dataset_subsets.values()]
    )
    for subset, dataset in dataset_subsets.items():
        for row in dataset:
            row = dict(row)
            split = "dev" if random.random() < 0.5 else "test"
            for i in range(1, 4):
                if f"belief_elicit_{i}" not in row:
                    continue
                records.append(
                    Record(
                        input=[
                            Message(
                                role="system", content="You are a helpful assistant."
                            ),
                            Message(role="user", content=row[f"belief_elicit_{i}"]),
                        ],
                        id=f"{row['task_id']}_belief_elicit_{i}",
                        other_fields={
                            **{k: row[k] for k in common_keys},
                            "subset": subset,
                            "pressure": "low",
                            "split": split,
                        },
                    )
                )
                records.append(
                    Record(
                        input=[
                            Message(role="system", content=row["system_prompt"]),
                            Message(role="user", content=row["user_prompt"]),
                        ],
                        id=f"{row['task_id']}_pressure",
                        other_fields={
                            **row,
                            "subset": subset,
                            "pressure": "high",
                            "split": split,
                        },
                    )
                )

    return Dataset.from_records(records)


def main(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    dataset = load_mask_dataset()
    raw = label_dataset(dataset, max_concurrent=200)
    raw_dev = raw.filter(lambda x: x.split == "dev")
    raw_test = raw.filter(lambda x: x.split == "test")
    raw_dev.save_to(EVALS_DIR / "mask_samples_raw.jsonl")
    raw_test.save_to(TEST_EVALS_DIR / "mask_samples_raw.jsonl")

    balanced_dev = subsample_balanced_subset(raw_dev)
    balanced_test = subsample_balanced_subset(raw_test)

    balanced_dev.save_to(EVALS_DIR / "mask_samples_balanced.jsonl")
    balanced_test.save_to(TEST_EVALS_DIR / "mask_samples_balanced.jsonl")


if __name__ == "__main__":
    main(seed=42)
