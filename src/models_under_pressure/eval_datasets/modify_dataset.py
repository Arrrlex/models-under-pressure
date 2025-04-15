from pathlib import Path
from typing import Any, List, NamedTuple, Optional

from models_under_pressure.config import DATA_DIR, EVAL_DATASETS_RAW, TEST_DATASETS_RAW
from models_under_pressure.eval_datasets.label_dataset import LabelledDataset
from models_under_pressure.eval_datasets.mts_dataset import get_mts_samples_by_ids
from models_under_pressure.interfaces.dataset import Dataset, Message
from models_under_pressure.utils import async_map, call_llm_async


class PromptModification(NamedTuple):
    original: str
    modified: Optional[str]


async def modify_single_system_prompt(
    messages: List[Message],
    model: str = "gpt-4o",
) -> PromptModification:
    """
    Modify a single system prompt to include deployment context while maintaining tool information.

    Args:
        messages: List of messages in the conversation
        model: The LLM model to use for modifying the prompt

    Returns:
        A tuple containing the original and modified system prompts
    """
    # Find the system message
    system_message = next((msg for msg in messages if msg.role == "system"), None)
    if system_message is None:
        return PromptModification(original="", modified=None)

    # Format the conversation for context
    conversation_context = "\n".join(
        [
            f"{msg.role}: {msg.content}"
            for msg in messages
            if msg.role != "system"  # Exclude the system message itself
        ]
    )

    # Modify the system prompt
    modification_messages = [
        {
            "role": "system",
            "content": "You are an expert in writing system prompts for AI assistants. Your task is to modify the given system prompt to include deployment context while maintaining all tool information and functionality. Keep the system prompt as concise as possible. Consider the full conversation context when adding deployment context.",
        },
        {
            "role": "user",
            "content": f"""Please modify this system prompt to include deployment context while maintaining all tool information and functionality. Consider the following conversation context when adding deployment context:

Conversation context:
{conversation_context}

Current system prompt:
{system_message.content}""",
        },
    ]

    try:
        modified_prompt = await call_llm_async(modification_messages, model)
        return PromptModification(
            original=system_message.content, modified=modified_prompt
        )
    except Exception as e:
        print(f"Error modifying prompt: {e}")
        return PromptModification(original=system_message.content, modified=None)


async def modify_system_prompts(
    dataset: LabelledDataset,
    output_path: Path,
    model: str = "gpt-4o",
    max_concurrent: int = 5,
) -> LabelledDataset:
    """
    Modify system prompts in the dataset to include deployment context while maintaining tool information.

    Args:
        dataset: The input dataset containing conversations with system prompts
        output_path: Path where to save the modified dataset
        model: The LLM model to use for modifying prompts
        max_concurrent: Maximum number of concurrent LLM calls

    Returns:
        The modified dataset with updated system prompts
    """
    # Convert dataset to records for processing
    records = dataset.to_records()

    # Process all records concurrently using async_map
    prompt_mods = await async_map(
        func=modify_single_system_prompt,
        items=[{"messages": record.input, "model": model} for record in records],
        max_concurrent=max_concurrent,
        with_pbar=True,
    )

    # Process results
    modified_inputs = []
    modified_other_fields = {k: [] for k in dataset.other_fields.keys()}
    original_prompts = []
    modified_prompts = []

    for record, prompt_mod in zip(records, prompt_mods):
        messages = record.input
        original_prompts.append(prompt_mod.original)
        modified_prompts.append(prompt_mod.modified)

        if prompt_mod.modified is None:
            print(f"Warning: Failed to modify prompt for record {record.id}")
            modified_inputs.append(messages)
        else:
            # Replace the system message with the modified one
            new_messages = [
                msg
                for msg in messages
                if not isinstance(msg, str) and msg.role != "system"
            ]
            new_messages.insert(0, Message(role="system", content=prompt_mod.modified))
            modified_inputs.append(new_messages)

        # Copy other fields
        for k, v in record.other_fields.items():
            modified_other_fields[k].append(v)

    # Add the prompt fields
    modified_other_fields["original_system_prompts"] = original_prompts
    modified_other_fields["modified_system_prompts"] = modified_prompts

    # Create new dataset with modified prompts
    modified_dataset = LabelledDataset(
        inputs=modified_inputs, ids=dataset.ids, other_fields=modified_other_fields
    )

    # Save the modified dataset
    modified_dataset.save_to(output_path, overwrite=True)
    return modified_dataset


def combine_datasets(
    mts_dataset: LabelledDataset, samples_path: Path, output_path: Path
) -> LabelledDataset:
    """
    Combine fields from the MTS dataset with a samples dataset, using the samples dataset as the base.

    Args:
        mts_dataset: The original MTS dataset with labels
        samples_path: Path to the samples dataset file
        output_path: Path where to save the combined dataset

    Returns:
        The combined dataset
    """
    if not samples_path.exists():
        samples = get_mts_samples_by_ids(mts_dataset.ids)  # type: ignore
        samples.save_to(samples_path)
    else:
        samples = Dataset.load_from(samples_path)
    print(f"Loaded {len(samples.ids)} samples")

    # Create a mapping from id to index for both datasets
    dataset_id_to_idx = {id: idx for idx, id in enumerate(mts_dataset.ids)}
    samples_id_to_idx = {id: idx for idx, id in enumerate(samples.ids)}

    # Find common ids
    common_ids = set(mts_dataset.ids) & set(samples.ids)

    # Create new other_fields by combining fields from both datasets
    new_other_fields = {}

    # First add all fields from samples dataset
    for field_name, field_values in samples.other_fields.items():
        new_other_fields[field_name] = field_values

    # Then add fields from original dataset, maintaining order based on sample IDs
    for field_name, field_values in mts_dataset.other_fields.items():
        if field_name not in new_other_fields:
            # Create a new list for this field with the same length as samples
            new_values: List[Any] = [None] * len(samples.ids)  # type: ignore
            # Fill in values for common IDs
            for id in common_ids:
                sample_idx = samples_id_to_idx[id]
                dataset_idx = dataset_id_to_idx[id]
                new_values[sample_idx] = field_values[dataset_idx]
            new_other_fields[field_name] = new_values

    # Create a new LabelledDataset with combined fields
    combined_dataset = LabelledDataset(
        inputs=samples.inputs, ids=samples.ids, other_fields=new_other_fields
    )

    print(f"Combined dataset size: {len(combined_dataset)}")
    print(f"Original dataset size: {len(mts_dataset)}")
    print(f"Samples dataset size: {len(samples)}")
    print(f"Number of common IDs: {len(common_ids)}")

    # Save the combined dataset
    combined_dataset.save_to(output_path, overwrite=True)
    return combined_dataset


if __name__ == "__main__":
    import asyncio

    dataset_name = "toolace"

    if dataset_name == "mts":
        dataset = LabelledDataset.load_from(EVAL_DATASETS_RAW["mts"])
        samples_path = DATA_DIR / "temp/mts_updated.jsonl"
        output_path = DATA_DIR / "temp/mts_combined.jsonl"

        # combine_datasets(dataset, samples_path, output_path)
        # print(Dataset.load_from(samples_path).ids)

        # dataset = LabelledDataset.load_from(output_path)
        # print(dataset.other_fields["labels"])
        print(len(dataset.ids))

    if dataset_name == "mts_test":
        dataset = LabelledDataset.load_from(TEST_DATASETS_RAW["mts"])
        samples_path = DATA_DIR / "temp/mts_test_updated.jsonl"
        output_path = DATA_DIR / "temp/mts_test_combined.jsonl"

        # combine_datasets(dataset, samples_path, output_path)
        # print(Dataset.load_from(samples_path).ids)

    if dataset_name == "toolace":
        dataset = LabelledDataset.load_from(EVAL_DATASETS_RAW["toolace"])[:20]

        # Test with a single sample first
        sample = dataset.sample(1)
        record = sample.to_records()[0]
        prompt_mod = asyncio.run(modify_single_system_prompt(record.input))

        print("\nOriginal system prompt:")
        print(prompt_mod.original)

        print("\nModified system prompt:")
        print(prompt_mod.modified)

        # If the single sample looks good, process the whole dataset
        output_path = DATA_DIR / "temp/toolace_modified.jsonl"
        modified_dataset = asyncio.run(modify_system_prompts(dataset, output_path))

        # Print a sample from the modified dataset
        random_item = modified_dataset.sample(1)
        print("\nSample from modified dataset:")
        print(random_item.to_records()[0])
