#!/usr/bin/env python3
"""
Script to convert generation baseline results to aggregated format.

This script takes generation baseline results and converts them to the same format
as continuation baseline results, with token counts calculated from actual input
conversations and output responses.

Usage:
    python convert_generation_to_continuation_format.py <model_name>

Example:
    python convert_generation_to_continuation_format.py gemma-12b
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from models_under_pressure.config import LOCAL_MODELS, RESULTS_DIR, TEST_DATASETS
from models_under_pressure.experiments.evaluate_probes import calculate_metrics
from models_under_pressure.interfaces.dataset import LabelledDataset, Message


def count_tokens_for_text(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], text: str
) -> int:
    """Count tokens for a given text using the tokenizer."""
    # Tokenize the text and count tokens
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    attention_mask = tokens["attention_mask"]
    # Convert to tensor if needed and sum
    if isinstance(attention_mask, torch.Tensor):
        return int(attention_mask.sum().item())
    else:
        # Convert to tensor first
        return int(torch.tensor(attention_mask).sum().item())


def count_tokens_for_dialogue(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    messages: List[Message],
) -> int:
    """Count tokens for a dialogue using the chat template."""
    # Convert messages to format expected by apply_chat_template
    message_dicts = [msg.model_dump() for msg in messages]

    # Apply chat template to get formatted text
    formatted_text = tokenizer.apply_chat_template(
        message_dicts, tokenize=False, add_generation_prompt=False
    )

    # Ensure we have a string
    if not isinstance(formatted_text, str):
        raise ValueError(
            f"Expected string from apply_chat_template, got {type(formatted_text)}"
        )

    # Count tokens for the formatted text
    return count_tokens_for_text(tokenizer, formatted_text)


def create_messages_from_prompt_config_and_conversation(
    user_prompt: str, system_prompt: str | None = None, conversation: str = ""
) -> List[Message]:
    """Create a list of messages from prompt configuration and actual conversation."""
    messages = []

    if system_prompt is not None:
        messages.append(Message(role="system", content=system_prompt))

    # Format the user prompt with the actual conversation
    formatted_user_prompt = user_prompt.format(conversation=conversation)
    messages.append(Message(role="user", content=formatted_user_prompt))

    return messages


def load_conversations_from_dataset(dataset_name: str) -> Dict[str, str]:
    """Load conversations from the dataset and return a mapping from ID to conversation."""

    # Infer dataset path from dataset name using TEST_DATASETS
    if dataset_name not in TEST_DATASETS:
        print(f"  Warning: Dataset '{dataset_name}' not found in TEST_DATASETS")
        print(f"  Available datasets: {list(TEST_DATASETS.keys())}")
        print("  Will use empty conversations for token counting")
        return {}

    dataset_path = TEST_DATASETS[dataset_name]
    print(f"  Loading dataset '{dataset_name}' from {dataset_path}...")

    try:
        dataset = LabelledDataset.load_from(dataset_path)

        # Create mapping from ID to conversation
        id_to_conversation = {}

        for id_, input_ in zip(dataset.ids, dataset.inputs):
            # Convert input to conversation string using same logic as generation.py format_conversation
            if isinstance(input_, str):
                conversation = input_
            elif isinstance(input_, list):
                # Handle list of messages - format exactly like generation.py
                formatted = []
                for msg in input_:
                    if hasattr(msg, "role") and hasattr(msg, "content"):
                        role = msg.role.upper()
                        content = msg.content
                        formatted.append(f"{role}\n{content}")
                    elif isinstance(msg, dict):
                        role = msg.get("role", "").upper()
                        content = msg.get("content", "")
                        formatted.append(f"{role}\n{content}")
                    else:
                        formatted.append(str(msg))
                conversation = "\n\n".join(formatted)
            else:
                conversation = str(input_)

            id_to_conversation[id_] = conversation

        print(f"  Loaded {len(id_to_conversation)} conversations")
        return id_to_conversation

    except Exception as e:
        print(
            f"  Warning: Failed to load dataset '{dataset_name}' from {dataset_path}: {e}"
        )
        print("  Will use empty conversations for token counting")
        return {}


def convert_generation_result_to_aggregated_format(
    generation_result: dict,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    model_name: str,
) -> dict:
    """Convert a single generation baseline result to aggregated format."""

    # Extract data from generation result
    ids = generation_result["ids"]
    accuracy = generation_result["accuracy"]
    labels = generation_result["labels"]
    ground_truth = generation_result["ground_truth"]
    ground_truth_scale_labels = generation_result.get("ground_truth_scale_labels")
    dataset_name = generation_result["dataset_name"]
    dataset_path = generation_result[
        "dataset_path"
    ]  # Keep for reference, but use TEST_DATASETS
    max_samples = generation_result.get("max_samples")
    full_responses = generation_result["full_response"]
    scores = generation_result["scores"]
    prompt_config = generation_result["prompt_config"]

    print(f"Processing {len(ids)} samples for dataset {dataset_name}...")

    # Load the original dataset to get binary labels for metric recalculation
    original_dataset = None
    if dataset_name in TEST_DATASETS:
        try:
            original_dataset = LabelledDataset.load_from(TEST_DATASETS[dataset_name])
            print(f"  Loaded original dataset with {len(original_dataset)} samples")
        except Exception as e:
            print(f"  Warning: Failed to load original dataset: {e}")
            print("  Will use existing ground_truth for metrics")
    else:
        print(f"  Warning: Dataset '{dataset_name}' not found in TEST_DATASETS")
        print("  Will use existing ground_truth for metrics")

    # Recalculate metrics using original dataset binary labels if available
    if original_dataset is not None:
        try:
            # Get binary labels from original dataset
            original_binary_labels = original_dataset.labels_numpy()

            # Ensure we have the same number of samples
            if len(original_binary_labels) == len(scores):
                # Recalculate metrics using calculate_metrics function
                recalculated_metrics = calculate_metrics(
                    y_true=original_binary_labels, y_pred=np.array(scores), fpr=0.01
                )

                # Update accuracy with recalculated value
                accuracy = recalculated_metrics["accuracy"]

                print("  Recalculated metrics:")
                print(f"    AUROC: {recalculated_metrics['auroc']:.4f}")
                print(f"    Accuracy: {recalculated_metrics['accuracy']:.4f}")
                print(f"    TPR at FPR=0.01: {recalculated_metrics['tpr_at_fpr']:.4f}")

                # Update ground_truth with original binary labels
                ground_truth = original_binary_labels.tolist()
            else:
                print(
                    f"  Warning: Sample count mismatch - original: {len(original_binary_labels)}, scores: {len(scores)}"
                )
                print("  Using existing ground_truth for metrics")
        except Exception as e:
            print(f"  Warning: Failed to recalculate metrics: {e}")
            print("  Using existing ground_truth for metrics")

    # Load actual conversations from the dataset using TEST_DATASETS
    id_to_conversation = load_conversations_from_dataset(dataset_name)

    # Calculate token counts for each sample
    token_counts = []

    for i, (sample_id, response, score) in enumerate(zip(ids, full_responses, scores)):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(ids)} samples")

        # Get the actual conversation for this sample
        conversation = id_to_conversation.get(sample_id, "")

        # For input tokens, we construct the prompt from prompt_config with actual conversation
        input_messages = create_messages_from_prompt_config_and_conversation(
            user_prompt=prompt_config["user_prompt"],
            system_prompt=prompt_config.get("system_prompt"),
            conversation=conversation,
        )

        # Count input tokens
        input_tokens = count_tokens_for_dialogue(tokenizer, input_messages)

        # Count output tokens
        output_tokens = count_tokens_for_text(tokenizer, response)

        # Total tokens = input + output
        total_tokens = input_tokens + output_tokens
        token_counts.append(total_tokens)

    # Create aggregated prompt configuration (following continuation baseline format)
    aggregated_prompt = {
        "high_stakes_completion": "",
        "low_stakes_completion": "",
        "system_prompt": prompt_config.get("system_prompt"),
        "user_prompt": prompt_config["user_prompt"],
        "conversation_input_key": prompt_config.get(
            "conversation_input_key", "user_prompt"
        ),
    }

    # Convert scores to high/low stakes scores (dummy values based on the single score)
    high_stakes_scores = [
        score if label == 1 else 1.0 - score for score, label in zip(scores, labels)
    ]
    low_stakes_scores = [
        1.0 - score if label == 1 else score for score, label in zip(scores, labels)
    ]

    # Create dummy log likelihoods (not available from generation baseline)
    high_stakes_log_likelihoods = [
        np.log(max(score, 1e-10)) for score in high_stakes_scores
    ]
    low_stakes_log_likelihoods = [
        np.log(max(score, 1e-10)) for score in low_stakes_scores
    ]

    # Create the aggregated format result
    aggregated_result = {
        "ids": ids,
        "accuracy": accuracy,
        "labels": labels,
        "ground_truth": ground_truth,
        "ground_truth_scale_labels": ground_truth_scale_labels,
        "dataset_name": dataset_name,
        "dataset_path": str(
            TEST_DATASETS.get(dataset_name, dataset_path)
        ),  # Use TEST_DATASETS path
        "model_name": model_name,
        "max_samples": max_samples,
        "timestamp": generation_result.get("timestamp"),
        "high_stakes_scores": high_stakes_scores,
        "low_stakes_scores": low_stakes_scores,
        "high_stakes_log_likelihoods": high_stakes_log_likelihoods,
        "low_stakes_log_likelihoods": low_stakes_log_likelihoods,
        "token_counts": token_counts,
        "prompt_config": aggregated_prompt,
    }

    return aggregated_result


def main():
    parser = argparse.ArgumentParser(
        description="Convert generation baseline results to aggregated format"
    )
    parser.add_argument("model_name", help="Model name (e.g., 'gemma-12b', 'llama-8b')")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=RESULTS_DIR / "baselines" / "generation",
        help="Directory containing generation baseline results",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RESULTS_DIR / "baselines" / "generation",
        help="Directory to save aggregated format results",
    )

    args = parser.parse_args()

    # Validate model name
    if args.model_name not in LOCAL_MODELS:
        print(f"Error: Model '{args.model_name}' not found in LOCAL_MODELS")
        print(f"Available models: {list(LOCAL_MODELS.keys())}")
        sys.exit(1)

    model_path = LOCAL_MODELS[args.model_name]
    print(f"Converting results for model: {args.model_name} ({model_path})")

    # Load tokenizer
    print(f"Loading tokenizer for {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Find all generation baseline result files for this model
    pattern = f"{model_path.split('/')[-1]}_*_generation_baseline.jsonl"
    input_files = list(args.input_dir.glob(pattern))

    if not input_files:
        print(f"No generation baseline files found matching pattern: {pattern}")
        print(f"Searched in directory: {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} generation baseline files:")
    for file in input_files:
        print(f"  - {file.name}")

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    output_file = args.output_dir / f"generation_baseline_{args.model_name}.jsonl"

    # Process each file and convert to aggregated format
    converted_results = []

    for input_file in input_files:
        print(f"\nProcessing {input_file.name}...")

        # Read the generation baseline results
        with open(input_file, "r") as f:
            lines = f.readlines()

        if not lines:
            print(f"  Warning: {input_file.name} is empty")
            continue

        # Process the last (most recent) result in the file
        generation_result = json.loads(lines[-1])

        # Convert to aggregated format
        aggregated_result = convert_generation_result_to_aggregated_format(
            generation_result, tokenizer, model_path
        )

        converted_results.append(aggregated_result)
        print(f"  Converted {len(generation_result['ids'])} samples")
        print(
            f"  Average tokens per sample: {np.mean(aggregated_result['token_counts']):.1f}"
        )

    # Write all results to output file
    print(f"\nWriting {len(converted_results)} dataset results to {output_file}")
    with open(output_file, "w") as f:
        for result in converted_results:
            f.write(json.dumps(result) + "\n")

    print("Conversion completed successfully!")

    # Print summary
    total_samples = sum(len(result["ids"]) for result in converted_results)
    all_token_counts = []
    for result in converted_results:
        all_token_counts.extend(result["token_counts"])

    print("\nSummary:")
    print(f"  Total datasets: {len(converted_results)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Average tokens per sample: {np.mean(all_token_counts):.1f}")
    print(f"  Token count range: [{min(all_token_counts)}, {max(all_token_counts)}]")


if __name__ == "__main__":
    main()
