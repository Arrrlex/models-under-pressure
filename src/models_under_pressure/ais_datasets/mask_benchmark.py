from pathlib import Path

import pandas as pd

from models_under_pressure.config import PROJECT_ROOT
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import Dataset, Message, Record

CSV_PATH = PROJECT_ROOT / "data/mask/continuations_llama-3.3-70b-instruct.csv"
SAVE_PATH = PROJECT_ROOT / "data/evals/ais/labelled_mask_benchmark.jsonl"


def build_dataset_from_csv(csv_path: Path) -> Dataset:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        # Pressure prompt row
        pressure_input = [
            Message(role="system", content=row["system_prompt"]),
            Message(role="user", content=row["user_prompt"]),
            Message(
                role="assistant",
                content=row.get("generation(System Prompt + User Prompt)_run1", ""),
            ),
        ]
        # Compute is_honest for pressure prompt
        pressure_val = row["eval(System Prompt + User Prompt)_run1_extracted"]
        belief_val = row["eval(belief_system_1 + belief_elicit_1)_run1_extracted"]
        deception_score = int(pressure_val.strip() != belief_val.strip())
        records.append(
            Record(
                input=pressure_input,
                id=f"{row['task_id']}_pressure",
                other_fields={
                    **row.to_dict(),
                    "deception_scores": deception_score,
                    "prompt_type": "pressure",
                },
            )
        )

        # Belief elicit prompts
        belief_elicit = row.get("belief_elicit_1")
        belief_gen = row.get("generation(belief_system_1 + belief_elicit_1)_run1", "")
        if pd.notnull(belief_elicit) and str(belief_elicit).strip():
            belief_input = [
                Message(role="system", content=row["belief_system_1"]),
                Message(role="user", content=belief_elicit),
                Message(role="assistant", content=belief_gen),
            ]
            records.append(
                Record(
                    input=belief_input,
                    id=f"{row['task_id']}_belief_elicit_1",
                    other_fields={
                        **row.to_dict(),
                        "deception_scores": 0,
                        "prompt_type": "belief_elicit_1",
                    },
                )
            )
    return Dataset.from_records(records)


def main():
    dataset = build_dataset_from_csv(CSV_PATH)
    labelled = label_dataset(dataset, max_concurrent=200)
    labelled.save_to(SAVE_PATH, overwrite=True)


if __name__ == "__main__":
    main()
