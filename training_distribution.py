import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from models_under_pressure.interfaces.dataset import LabelledDataset, to_dialogue
from models_under_pressure.config import LOCAL_MODELS

# Default values
DEFAULT_DATASET_PATH = Path("data/training/prompts_4x/train.jsonl")
DEFAULT_MODEL_NAME = LOCAL_MODELS["gemma-12b"]


def tokenize(tokenizer, input):
    dialogue = to_dialogue(input)
    input_dicts = [[d.model_dump() for d in dialogue]]
    input_str = tokenizer.apply_chat_template(
        input_dicts,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(input_str)
    return tokens["input_ids"][0]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token length distribution in a dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the dataset (.jsonl or .csv)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name or path for tokenizer",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    dataset = LabelledDataset.load_from(Path(args.dataset_path))
    print(f"Loaded {len(dataset)} samples.")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Tokenizing samples...")
    token_lengths = [len(tokenize(tokenizer, input)) for input in dataset.inputs]
    token_lengths = np.array(token_lengths)

    print(f"\nNumber of samples: {len(token_lengths)}")
    print(f"Longest sample: {token_lengths.max()} tokens")
    for k in [99.99, 99.9, 99.5, 99, 97, 95, 90, 80]:
        print(
            f"{k}% of samples are shorter than: {int(np.percentile(token_lengths, k))} tokens"
        )


if __name__ == "__main__":
    main()
