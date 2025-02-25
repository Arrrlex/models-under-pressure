from datasets import load_dataset

# TODO Randomly sample 100 cases from both categories and annotate them for high-stakes and low-stakes


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


if __name__ == "__main__":
    dataset = load_anthropic_dataset(split="train")

    print(len(dataset))
    print(dataset[0])
