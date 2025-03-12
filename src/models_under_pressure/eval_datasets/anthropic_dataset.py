from collections import defaultdict

from datasets import load_dataset

from models_under_pressure.config import EVAL_DATASETS_BALANCED, EVAL_DATASETS_RAW
from models_under_pressure.eval_datasets.label_dataset import create_eval_dataset
from models_under_pressure.interfaces.dataset import Dataset, Message


def parse_messages(text: str) -> list[Message]:
    text_parts = text.split("\n\n")
    messages = []
    current_message = None
    current_role = None

    for part in text_parts:
        if part.startswith("Human:"):
            # Save previous message if exists
            if current_message is not None:
                assert current_role is not None
                messages.append(Message(role=current_role, content=current_message))
            current_message = part[len("Human:") :].strip()
            current_role = "user"
        elif part.startswith("Assistant:"):
            if current_message is not None:
                assert current_role is not None
                messages.append(Message(role=current_role, content=current_message))
            current_message = part[len("Assistant:") :].strip()
            current_role = "assistant"
        elif len(part.strip()) > 0:
            if current_message is not None:
                # Append to existing message with a newline
                current_message += "\n\n" + part.strip()
            else:
                # Handle system message or unknown start
                current_message = part.strip()
                current_role = "system"

    # Add the final message
    if current_message is not None:
        assert current_role is not None
        messages.append(Message(role=current_role, content=current_message))

    return messages


def load_anthropic_raw_data(split: str = "train") -> Dataset:
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    ids = []
    inputs = []
    other_fields = defaultdict(list)

    for ix, sample in enumerate(dataset):  # type: ignore
        for field in ["chosen", "rejected"]:
            # Extract messages
            text = sample[field]  # type: ignore
            messages = parse_messages(text)

            ids.append(f"{split}_{ix}_{field}")
            inputs.append(messages)
            other_fields["split"].append(split)
            other_fields["category"].append(field)
            other_fields["index"].append(ix)

    return Dataset(
        inputs=inputs,
        ids=ids,
        other_fields=other_fields,
    )


def create_anthropic_dataset(num_samples: int = 100, recompute: bool = False):
    dataset = load_anthropic_raw_data()
    dataset = dataset.sample(num_samples)
    return create_eval_dataset(
        dataset,
        raw_output_path=EVAL_DATASETS_RAW["anthropic"],
        balanced_output_path=EVAL_DATASETS_BALANCED["anthropic"],
        recompute=recompute,
    )


if __name__ == "__main__":
    create_anthropic_dataset(num_samples=100)
