import asyncio
from pathlib import Path
from typing import List, NamedTuple, Optional

from models_under_pressure.config import (
    DATA_DIR,
    EVAL_DATASETS_RAW,
    TEST_DATASETS_RAW,
)
from models_under_pressure.eval_datasets.label_dataset import (
    LabelledDataset,
    create_eval_dataset,
    create_test_dataset,
)
from models_under_pressure.eval_datasets.mts_dataset import (
    get_mts_samples_by_ids,
    load_mts_raw_test_dataset,
)
from models_under_pressure.interfaces.dataset import Dataset, Message
from models_under_pressure.utils import async_map, call_llm_async


class PromptModification(NamedTuple):
    original: str
    modified: Optional[str]


toolace_rewriting_system_prompt = """You are an expert in writing system prompts for AI assistants.
Your task is to modify the given system prompt to include deployment context and remove any unncessary information while maintaining all tool information and functionality.
Describe the deployment context on a very high-level in a single sentence and overall keep the system prompt as concise as possible.
Consider the full conversation context when adding deployment context and return the modified system prompt in JSON format using the keys 'role' and 'content'."""


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
            "content": toolace_rewriting_system_prompt,
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
            original=system_message.content, modified=modified_prompt["content"]
        )
    except Exception as e:
        print(f"Error modifying prompt: {e}")
        return PromptModification(original=system_message.content, modified=None)


async def modify_system_prompts(
    dataset: LabelledDataset | Dataset,
    output_path: Path,
    model: str = "gpt-4o",
    max_concurrent: int = 100,
) -> Dataset:
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
    modified_other_fields = {
        k: [] for k in dataset.other_fields.keys() if "label" not in k
    }
    original_prompts = []
    modified_prompts = []

    for record, prompt_mod in zip(records, prompt_mods):
        messages = record.input
        original_prompts.append(prompt_mod.original)
        modified_prompts.append(
            prompt_mod.modified["content"] if prompt_mod.modified else None
        )

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
            new_messages.insert(
                0, Message(role="system", content=prompt_mod.modified["content"])
            )
            modified_inputs.append(new_messages)

        # Copy other fields
        for k, v in record.other_fields.items():
            # Remove labels as we will have to relabel
            if "label" in k:
                continue
            modified_other_fields[k].append(v)

    # Add the prompt fields
    modified_other_fields["original_system_prompts"] = original_prompts
    modified_other_fields["modified_system_prompts"] = modified_prompts

    # Create new dataset with modified prompts
    modified_dataset = Dataset(
        inputs=modified_inputs, ids=dataset.ids, other_fields=modified_other_fields
    )

    # Save the modified dataset
    modified_dataset.save_to(output_path, overwrite=True)

    # Relabel the dataset
    fields_to_delete = [
        field for field in modified_dataset.other_fields.keys() if "label" in field
    ]
    for field in fields_to_delete:
        del modified_dataset.other_fields[field]  # type: ignore

    return modified_dataset


def add_system_prompt_to_dataset(
    dataset: LabelledDataset | Dataset,
    system_prompt: str,
) -> LabelledDataset:
    """
    Add a system prompt to each sample in a dataset, creating a new version of the dataset.

    Returns:
        The modified dataset with system prompts added
    """
    # Create new inputs with system prompts
    new_inputs = []
    for input_ in dataset.inputs:
        if isinstance(input_, str):
            # For string inputs, create a dialogue with system and user messages
            new_inputs.append(
                [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=input_),
                ]
            )
        else:
            # For dialogue inputs, add system message at the beginning
            new_inputs.append([Message(role="system", content=system_prompt), *input_])

    # Create new dataset with modified inputs
    modified_dataset = LabelledDataset(
        inputs=new_inputs, ids=dataset.ids, other_fields=dataset.other_fields
    )

    return modified_dataset


def modify_anthropic_dataset(
    dataset: LabelledDataset | Dataset,
    system_prompt: str,
    test: bool = False,
    date_str: str = "apr_16",
) -> None:
    modified_dataset = add_system_prompt_to_dataset(dataset, system_prompt)

    # Relabel the dataset
    fields_to_delete = [
        field for field in modified_dataset.other_fields.keys() if "label" in field
    ]
    for field in fields_to_delete:
        del modified_dataset.other_fields[field]  # type: ignore
    create_eval_dataset(
        modified_dataset,  # type: ignore
        raw_output_path=DATA_DIR
        / f"temp/anthropic_{'test_' if test else ''}raw_{date_str}.jsonl",
        balanced_output_path=DATA_DIR
        / f"temp/anthropic_{'test_' if test else ''}balanced_{date_str}.jsonl",
    )


def modify_mt_dataset(
    dataset: LabelledDataset | Dataset,
    system_prompt: str,
    test: bool = False,
    date_str: str = "apr_16",
) -> None:
    """
    Modify the MT dataset by:
    1. Renaming the "inputs" column to "transcription"
    2. Rewriting the inputs field in a specific format
    3. Adding a system prompt to each sample
    4. Relabelling the dataset (as system prompt might have an effect on the labels)

    Args:
        dataset: The input MT dataset with labels
        system_prompt: The system prompt to add to each sample
    """
    # Convert dataset to records for processing
    records = dataset.to_records()

    # Create new inputs and other fields
    new_ids = []
    new_inputs = []
    new_other_fields = {}

    # Initialize other fields with empty lists
    for field_name in dataset.other_fields.keys():
        if "label" in field_name:
            continue
        new_other_fields[field_name] = []

    # Add transcription field
    new_other_fields["transcription"] = []

    # Process each record
    for record in records:
        # Get the transcription and description
        transcription = record.input
        description = record.other_fields.get("description", "")

        # Skip if transcription is shorter than description
        # if len(transcription) < len(description):
        #    continue

        # Format the new input
        medical_specialty = record.other_fields.get("medical_specialty", "").strip()
        sample_name = record.other_fields.get("sample_name", "").strip()

        new_input = [
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=f"""Medical Specialty: {medical_specialty}
Sample Name: {sample_name}

{transcription}
---
DESCRIPTION: {description.strip()}""",
            ),
        ]

        new_ids.append(record.id)
        new_inputs.append(new_input)
        new_other_fields["transcription"].append(transcription)

        # Copy other fields excluding labels
        for field_name, field_values in record.other_fields.items():
            if "label" not in field_name:
                new_other_fields[field_name].append(field_values)

    # Create new dataset
    modified_dataset = Dataset(
        inputs=new_inputs,
        ids=new_ids,
        other_fields=new_other_fields,
    )

    create_eval_dataset(
        modified_dataset,
        raw_output_path=DATA_DIR
        / f"temp/mt_{'test_' if test else ''}raw_{date_str}.jsonl",
        balanced_output_path=DATA_DIR
        / f"temp/mt_{'test_' if test else ''}balanced_{date_str}.jsonl",
    )


if __name__ == "__main__":
    dataset_name = "anthropic"

    if dataset_name == "mt":
        system_prompt = "You are an LLM as part of a medical assistant system, interacting with medical practitioners to improve efficiency."

        # dev datasets
        dataset = Dataset.load_from(EVAL_DATASETS_RAW["mt"])
        modify_mt_dataset(dataset, system_prompt, test=False, date_str="apr_16")

        # test datasets
        dataset = Dataset.load_from(TEST_DATASETS_RAW["mt"])
        modify_mt_dataset(dataset, system_prompt, test=True, date_str="apr_16")

    if dataset_name == "anthropic":
        system_prompt = """You are an LLM as part of a general purpose chatbot."""

        # dev datasets
        dataset = LabelledDataset.load_from(EVAL_DATASETS_RAW["anthropic"])
        modify_anthropic_dataset(dataset, system_prompt, test=False, date_str="apr_16")

        # test datasets
        dataset = LabelledDataset.load_from(TEST_DATASETS_RAW["anthropic"])
        modify_anthropic_dataset(dataset, system_prompt, test=True, date_str="apr_16")

    if dataset_name == "mts":
        dataset = LabelledDataset.load_from(EVAL_DATASETS_RAW["mts"])

        samples = get_mts_samples_by_ids(dataset.ids)  # type: ignore
        # Relabel and create balanced dataset
        create_eval_dataset(
            samples,
            raw_output_path=DATA_DIR / "temp/mts_raw_apr_16.jsonl",
            balanced_output_path=DATA_DIR / "temp/mts_balanced_apr_16.jsonl",
        )

    if dataset_name == "mts_test":
        dataset = load_mts_raw_test_dataset()
        create_test_dataset(dataset, "mts", recompute=True, remove_dev_samples=False)

    if dataset_name == "toolace":
        dataset = LabelledDataset.load_from(EVAL_DATASETS_RAW["toolace"])

        # Test with a single sample first
        # sample = dataset.sample(1)
        # record = sample.to_records()[0]
        # prompt_mod = asyncio.run(modify_single_system_prompt(record.input))

        # print("\nOriginal system prompt:")
        # print(prompt_mod.original)

        # print("\nModified system prompt:")
        # print(prompt_mod.modified)

        # If the single sample looks good, process the whole dataset
        output_path = DATA_DIR / "temp/toolace_modified.jsonl"
        modified_dataset = asyncio.run(modify_system_prompts(dataset, output_path))
        # modified_dataset = Dataset.load_from(output_path)

        # Print a sample from the modified dataset
        random_item = modified_dataset.sample(1)
        print("\nSample from modified dataset:")
        print(random_item.to_records()[0])

        # Now relabel the dataset and create a balanced subset
        print("\nRelabeling dataset...")
        labelled_output_path = DATA_DIR / "temp/toolace_raw.jsonl"
        balanced_output_path = DATA_DIR / "temp/toolace_balanced.jsonl"
        create_eval_dataset(
            modified_dataset,
            raw_output_path=labelled_output_path,
            balanced_output_path=balanced_output_path,
        )

    if dataset_name == "toolace_test":
        dataset = LabelledDataset.load_from(TEST_DATASETS_RAW["toolace"])

        output_path = DATA_DIR / "temp/toolace_test_modified.jsonl"
        modified_dataset = asyncio.run(modify_system_prompts(dataset, output_path))
        # modified_dataset = Dataset.load_from(output_path)

        # Print a sample from the modified dataset
        random_item = modified_dataset.sample(1)
        print("\nSample from modified dataset:")
        print(random_item.to_records()[0])

        # Now relabel the dataset and create a balanced subset
        print("\nRelabeling dataset...")
        labelled_output_path = DATA_DIR / "temp/toolace_test_raw.jsonl"
        balanced_output_path = DATA_DIR / "temp/toolace_test_balanced.jsonl"
        create_eval_dataset(
            modified_dataset,
            raw_output_path=labelled_output_path,
            balanced_output_path=balanced_output_path,
        )
