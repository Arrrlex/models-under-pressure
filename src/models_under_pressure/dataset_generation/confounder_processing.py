import asyncio
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from models_under_pressure.config import (
    global_settings,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.utils import call_llm_async

High_stakes_words_list = [
    "Urgent",
    "Immediate",
    "Critical",
    "Sensitive",
    "Pressing",
    "Emergency",
    "At once",
    "Without delay",
    "Rapid response",
    "Major",
    "Significant",
    "Crucial",
    "Vital",
    "Essential",
    "High priority",
    "Paramount",
    "Game-changing",
    "High impact",
    "Decisive",
    "Hazardous",
    "Dangerous",
    "Catastrophic",
    "Severe",
    "Dire",
    "Life-threatening",
    "Alarming",
    "Red flag",
    "Security breach",
    "Compromised",
    "Sensitive",
    "Classified",
    "Restricted",
    "Off the record",
    "Caution",
    "Alert",
    "Warning",
    "Threat",
    "Vulnerability",
    "Comply",
    "Mandatory",
    "Breach",
    "Violation",
]

Low_stakes_words_list = [
    "Non-urgent",
    "No rush",
    "Low priority",
    "Casual",
    "Minor",
    "Mild",
    "Marginal",
    "Trivial",
    "Insignificant",
    "Negligible",
    "Routine",
    "Optional",
    "Uncritical",
    "Inconsequential",
    "Safe",
    "Harmless",
    "Routine",
    "Low risk",
    "Stable",
    "Manageable",
    "Reversible",
    "Low impact",
    "Minimal disruption",
    "Just checking in",
    "Informal",
    "General",
    "Heads-up",
    "Not a big deal",
    "Simple",
]


async def neutralize_prompt_async(prompt: str) -> str:
    system_prompt = (
        "You're a helpful assistant that rewrites text by removing or neutralizing words "
        "that explicitly signal high or low stakes. "
        "Your goal is to preserve the overall meaning and context, but reduce obvious cues that "
        "signal emotional or risk-related intensity. Rewrite the input accordingly. You need not change input most of the times, only if it is very obvious and contains specific words from the list. You can also sometimes let the input be as it is if it the word is really needed for the sentence. Sometimes its okay."
    )

    user_prompt = (
        f"Original: {prompt}\n"
        f"List of high-stakes words: {', '.join(High_stakes_words_list)}\n"
        f"List of low-stakes words: {', '.join(Low_stakes_words_list)}\n"
        f"Provide your response in JSON format with a single key 'neutral_prompt' containing the neutralized text."
    )

    try:
        json_schema = {
            "name": "NeutralizedPrompt",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "neutral_prompt": {"type": "string"},
                },
                "required": ["neutral_prompt"],
                "additionalProperties": False,
            },
        }

        response = await call_llm_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=global_settings.DEFAULT_MODEL,
            json_schema=json_schema,
        )

        # Parse the JSON response
        response_content = response["neutral_prompt"].strip()
        return response_content
    except Exception as e:
        print(f"Error: {e}")
        return prompt


def neutralize_prompt(prompt: str) -> str:
    """Synchronous wrapper for the async function"""
    return asyncio.run(neutralize_prompt_async(prompt))


async def manipulate_stakes_async(prompt: str, is_high_stakes: bool) -> str:
    """
    Manipulates the prompt by adding words that contradict its stakes level.
    For high-stakes prompts, it adds low-stakes words.
    For low-stakes prompts, it adds high-stakes words.
    The meaning should remain the same.

    Args:
        prompt: The original prompt
        is_high_stakes: Whether the prompt is high stakes

    Returns:
        The manipulated prompt
    """
    system_prompt = (
        "You're a helpful assistant that rewrites text by strategically adding words "
        "that contradict the actual stakes level of the situation, while preserving the "
        "original meaning and context. "
        "For high-stakes situations, add low-stakes words. "
        "For low-stakes situations, add high-stakes words. "
        "The core meaning and actual stakes of the situation should remain unchanged - "
        "only the language should be manipulated."
    )

    stakes_level = "high-stakes" if is_high_stakes else "low-stakes"
    words_to_add = Low_stakes_words_list if is_high_stakes else High_stakes_words_list

    user_prompt = (
        f"Original {stakes_level} prompt: {prompt}\n"
        f"Words to strategically incorporate: {', '.join(words_to_add)}\n"
        f"Rewrite this {stakes_level} prompt to include some of the contradictory stakes words "
        f"while preserving the original meaning and actual stakes level. "
        f"Provide your response in JSON format with a single key 'manipulated_prompt' containing the rewritten text."
    )

    try:
        json_schema = {
            "name": "ManipulatedPrompt",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "manipulated_prompt": {"type": "string"},
                },
                "required": ["manipulated_prompt"],
                "additionalProperties": False,
            },
        }

        response = await call_llm_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=global_settings.DEFAULT_MODEL,
            json_schema=json_schema,
        )

        return response["manipulated_prompt"].strip()
    except Exception as e:
        print(f"Error in manipulate_stakes_async: {e}")
        return prompt


def manipulate_stakes(prompt: str, is_high_stakes: bool) -> str:
    """Synchronous wrapper for the async function"""
    return asyncio.run(manipulate_stakes_async(prompt, is_high_stakes))


def filter_dataset(dataset: LabelledDataset) -> LabelledDataset:
    """Filter the dataset to only include prompts that have scale labels more than 7 or less than 3 and scale label confidence is more than 7"""
    return dataset.filter(
        lambda x: x.other_fields["scale_labels"] is not None
        and x.other_fields["scale_label_confidence"] > 7
        and x.other_fields["scale_labels"] not in [4, 5, 6]
    )


# Apply labeling
tqdm.pandas()
# input_path = Path(
#     "/Users/urjapawar/Documents/refactor]/models-under-pressure/data/training/prompts_25_03_25_gpt-4o.jsonl"
# )
# neutralised_path = Path(
#     "/Users/urjapawar/Documents/refactor]/models-under-pressure/data/training/prompts_25_03_25_gpt-4o_neutralised.jsonl"
# )
# manipulated_path = Path(
#     "/Users/urjapawar/Documents/refactor]/models-under-pressure/data/training/prompts_25_03_25_gpt-4o_manipulated.jsonl"
# )

input_path = Path(
    "/Users/urjapawar/Documents/refactor]/models-under-pressure/data/results/debug/prompts_16_04_25_gpt-4o_balanced.jsonl"
)
neutralised_path = Path(
    "/Users/urjapawar/Documents/refactor]/models-under-pressure/data/results/debug/prompts_16_04_25_gpt-4o_balanced_neutralised.jsonl"
)
manipulated_path = Path(
    "/Users/urjapawar/Documents/refactor]/models-under-pressure/data/results/debug/prompts_16_04_25_gpt-4o_balanced_manipulated.jsonl"
)

df = pd.read_json(input_path, orient="records", lines=True)

tqdm.pandas()
# apply neutralization
neutralised_df = df.sample(frac=0.2)
neutralised_df["inputs"] = neutralised_df["inputs"].progress_apply(neutralize_prompt)
neutralised_df.to_json(neutralised_path, orient="records", lines=True)

# apply manipulation
manipulated_df = df.sample(frac=0.2)
manipulated_df["inputs"] = [
    manipulate_stakes(row["inputs"], row["labels"] == "high-stakes")
    for _, row in tqdm(manipulated_df.iterrows(), total=len(manipulated_df))
]
manipulated_df.to_json(manipulated_path, orient="records", lines=True)
