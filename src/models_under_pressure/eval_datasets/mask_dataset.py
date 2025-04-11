import random
from datasets import load_dataset
import numpy as np
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import Dataset, Record, Message
from models_under_pressure.utils import hf_login
from models_under_pressure.config import DATA_DIR

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
    labelled_dataset = label_dataset(dataset, model="gpt-4o")
    labelled_dataset.filter(lambda x: x.split == "dev").save_to(
        DATA_DIR / "evals" / "dev" / "mask_samples.jsonl"
    )
    labelled_dataset.filter(lambda x: x.split == "test").save_to(
        DATA_DIR / "evals" / "test" / "mask_samples.jsonl"
    )


if __name__ == "__main__":
    main(seed=42)
