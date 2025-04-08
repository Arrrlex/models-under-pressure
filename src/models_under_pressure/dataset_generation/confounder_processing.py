import asyncio

import pandas as pd
from tqdm.auto import tqdm

from models_under_pressure.config import (
    DEFAULT_MODEL,
    SYNTHETIC_DATASET_PATH,
)
from models_under_pressure.utils import call_llm_async

df = pd.read_json(SYNTHETIC_DATASET_PATH, orient="records", lines=True)

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
            model=DEFAULT_MODEL,
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


# Apply transformation
# tqdm.pandas()
df["neutral_prompt"] = df["prompt"].apply(neutralize_prompt)

# Save to new CSV
df.to_csv(
    str(SYNTHETIC_DATASET_PATH).replace(".jsonl", "_neutralised.csv"), index=False
)


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
            model=DEFAULT_MODEL,
            json_schema=json_schema,
        )

        return response["manipulated_prompt"].strip()
    except Exception as e:
        print(f"Error in manipulate_stakes_async: {e}")
        return prompt


def manipulate_stakes(prompt: str, is_high_stakes: bool) -> str:
    """Synchronous wrapper for the async function"""
    return asyncio.run(manipulate_stakes_async(prompt, is_high_stakes))


df = pd.read_json(SYNTHETIC_DATASET_PATH, orient="records", lines=True)
df["boolean_label"] = df["scale_labels"].apply(lambda x: x == "high-stakes")
# Example usage:
tqdm.pandas()
df["manipulated_prompt"] = [
    manipulate_stakes(row["prompt"], row["boolean_label"])
    for _, row in tqdm(df.iterrows(), total=len(df))
]
df.to_csv(
    str(SYNTHETIC_DATASET_PATH).replace(".jsonl", "_manipulated.csv"), index=False
)
