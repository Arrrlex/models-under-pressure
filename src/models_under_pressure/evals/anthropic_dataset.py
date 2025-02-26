import csv
import json
import random

from datasets import load_dataset

from models_under_pressure.config import EVALS_DIR


def load_anthropic_dataset(split: str = "train") -> list[dict]:
    dataset = load_dataset("Anthropic/hh-rlhf")
    data_dicts = []
    for ix, sample in enumerate(dataset[split]):  # type: ignore
        for field in ["chosen", "rejected"]:
            # Extract messages
            text = sample[field]  # type: ignore
            text_parts = text.split("\n\n")
            messages = []
            current_message = None
            current_role = None

            for part in text_parts:
                if part.startswith("Human:"):
                    # Save previous message if exists
                    if current_message is not None:
                        messages.append(
                            {"role": current_role, "content": current_message.strip()}
                        )
                    current_message = part[len("Human:") :].strip()
                    current_role = "user"
                elif part.startswith("Assistant:"):
                    if current_message is not None:
                        messages.append(
                            {"role": current_role, "content": current_message.strip()}
                        )
                    current_message = part[len("Assistant:") :].strip()
                    current_role = "assistant"
                elif len(part.strip()) > 0:
                    if current_message is not None:
                        # Append to existing message with a newline
                        current_message += "\n\n" + part.strip()
                    else:
                        # Handle system message or unknown start
                        print(
                            "Warning: Unknown message type. Interpreting as system message."
                        )
                        current_message = part.strip()
                        current_role = "system"
                        print(part)

            # Add the final message
            if current_message is not None:
                messages.append(
                    {"role": current_role, "content": current_message.strip()}
                )

            data_dict = {
                "id": f"{split}_{ix}_{field}",
                "index": ix,
                "split": split,
                "messages": messages,
                "category": field,
            }
            data_dicts.append(data_dict)
    return data_dicts


def generate_anthropic_samples(
    split: str = "train",
    num_samples: int = 100,
    file_name: str = "anthropic_samples_to_annotate.csv",
    overwrite: bool = False,
):
    dataset = load_anthropic_dataset(split=split)

    # Sample 100 unique indices
    unique_indices = list(set(d["index"] for d in dataset))
    sampled_indices = random.sample(unique_indices, k=num_samples)

    # Filter dataset to only include entries with sampled indices
    sampled_dataset = [d for d in dataset if d["index"] in sampled_indices]

    print(f"Sampled {len(sampled_dataset)} conversations")
    print(sampled_dataset[0])

    # Prepare data for CSV
    csv_data = []
    for item in sampled_dataset:
        csv_data.append(
            {
                "id": item["id"],
                "index": item["index"],
                "category": item["category"],
                "messages": json.dumps(
                    item["messages"]
                ),  # Convert messages list to JSON string
                "high_stakes": "",  # Empty column for manual annotation
            }
        )

    # Write to CSV
    output_file = EVALS_DIR / file_name
    if not EVALS_DIR.exists():
        EVALS_DIR.mkdir(parents=True, exist_ok=True)

    if not overwrite and output_file.exists():
        raise FileExistsError(
            f"File {output_file} already exists. Use overwrite=True to overwrite."
        )

    fieldnames = ["id", "index", "category", "messages", "high_stakes", "comment"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Wrote {len(csv_data)} rows to {output_file}")


if __name__ == "__main__":
    generate_anthropic_samples(split="train", num_samples=100)
