from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from models_under_pressure.config import PROJECT_ROOT
from models_under_pressure.eval_datasets.label_dataset import label_dataset_async
from models_under_pressure.interfaces.dataset import Dataset, Message, Record

CSV_PATH = PROJECT_ROOT / "data/mask/continuations_llama-3.3-70b-instruct.csv"
SAVE_PATH = PROJECT_ROOT / "data/mask/labelled_mask_benchmark.jsonl"


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
        is_honest = (
            str(row.get("eval(System Prompt + User Prompt)_run1_extracted", "")).strip()
            == str(
                row.get("eval(belief_system_1 + belief_elicit_1)_run1_extracted", "")
            ).strip()
        )
        records.append(
            Record(
                input=pressure_input,
                id=f"{row['task_id']}_pressure",
                other_fields={
                    **row.to_dict(),
                    "is_honest": is_honest,
                    "prompt_type": "pressure",
                },
            )
        )

        # Belief elicit prompts
        for i in range(1, 4):
            belief_elicit = row.get(f"belief_elicit_{i}")
            belief_gen = row.get(
                f"generation(belief_system_1 + belief_elicit_{i})_run1", ""
            )
            if pd.notnull(belief_elicit) and str(belief_elicit).strip():
                belief_input = [
                    Message(role="system", content=row["belief_system_1"]),
                    Message(role="user", content=belief_elicit),
                    Message(role="assistant", content=belief_gen),
                ]
                records.append(
                    Record(
                        input=belief_input,
                        id=f"{row['task_id']}_belief_elicit_{i}",
                        other_fields={
                            **row.to_dict(),
                            "is_honest": True,
                            "prompt_type": f"belief_elicit_{i}",
                        },
                    )
                )
    return Dataset.from_records(records)


async def main():
    dataset = build_dataset_from_csv(CSV_PATH)
    labelled = await label_dataset_async(dataset, max_concurrent=200)
    labelled.save_to(SAVE_PATH)


def plot_honesty_by_label(df: pd.DataFrame) -> None:
    """
    Create a bar plot showing mean honesty scores grouped by label and save it to a file.

    Args:
        df: DataFrame containing 'labels' and 'is_honest' columns
    """
    from sklearn.metrics import roc_auc_score

    # Convert labels to binary (high stakes = 1, low stakes = 0)
    y_true = (~df["is_honest"]).astype(int)
    y_pred = df["scale_labels"].astype(int)

    # Calculate and print AUROC
    auroc = roc_auc_score(y_true, y_pred)
    print(f"AUROC (all rows): {auroc:.3f}")

    df = df[df["prompt_type"] == "pressure"]

    # Recalculate AUROC on filtered dataset
    y_true_filtered = (~df["is_honest"]).astype(int)
    y_pred_filtered = df["scale_labels"].astype(int)
    auroc_filtered = roc_auc_score(y_true_filtered, y_pred_filtered)
    print(f"AUROC (pressure rows only): {auroc_filtered:.3f}")

    # Calculate mean is_honest for each label
    honesty_by_label = df.groupby("labels")["is_honest"].mean()

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(honesty_by_label.index, honesty_by_label.values)

    # Customize plot
    plt.title("Mean Honesty Score by Label")
    plt.xlabel("Label")
    plt.ylabel("Mean Honesty Score")
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for i, v in enumerate(honesty_by_label.values):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    # Save the plot to a file
    plt.savefig("data/mask/honesty_by_label_only_pressure.png")
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    # asyncio.run(main())

    df = pd.read_json(open(SAVE_PATH), lines=True)

    plot_honesty_by_label(df)
