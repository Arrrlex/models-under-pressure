from datasets import load_dataset
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import Dataset, Record, Message
from models_under_pressure.utils import hf_login
from models_under_pressure.config import DATA_DIR

hf_login()


def load_mask_dataset():
    dataset = load_dataset("cais/MASK", name="known_facts", split="test")
    records = []

    for row in dataset:
        row = dict(row)
        for i in range(1, 4):
            records.append(
                Record(
                    input=[
                        Message(role="system", content="You are a helpful assistant."),
                        Message(role="user", content=row[f"belief_elicit_{i}"]),
                    ],
                    id=f"{row['task_id']}_belief_elicit_{i}",
                    other_fields={
                        **row,
                        "subset": "known_facts",
                        "pressure": "low",
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
                    "subset": "known_facts",
                    "pressure": "high",
                },
            )
        )

    return Dataset.from_records(records)


def main():
    dataset = load_mask_dataset()
    labelled_dataset = label_dataset(dataset, model="gpt-4o")
    labelled_dataset.save_to(DATA_DIR / "dev" / "mask_dataset.jsonl")


if __name__ == "__main__":
    main()
