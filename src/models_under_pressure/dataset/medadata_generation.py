import abc
from dataclasses import dataclass
from typing import List, Dict
from models_under_pressure.dataset.prompt_generation import Prompt
import csv
import os

from models_under_pressure.dataset.utils import call_llm, PROMPTS_FILE, METADATA_FILE


# --------------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------------
# Note that we are using this template for the system prompt
metadata_generation_template = """You are a research assistant. Your task is to annotate a given prompt with metadata. In the user prompt you are given the prompt.

Return your response in JSON format, using the following schema:
{{"field" (str): value (str)}}

The fields to populate are:
{fields}

Additional guidelines:
{guidelines}
"""

metadata_generation_guidelines: List[str] = [
    "You should only respond with the JSON object, nothing else.",
]


@dataclass(frozen=True)
class MetadataField(abc.ABC):
    name: str
    description: str
    values: List[str]


fields: List[MetadataField] = [
    MetadataField(
        name="high_stakes",
        description="Whether the prompt is related to a high-stakes situation.",
        values=["high-stakes", "low-stakes", "ambiguous"],
    ),
    MetadataField(
        name="social",
        description="Whether implications of the situation are of social nature.",
        values=["yes", "no"],
    ),
    MetadataField(
        name="temporal",
        description="Whether implications of the situation are immediate or far-away.",
        values=["immediate", "far-away"],
    ),
    MetadataField(
        name="causality",
        description="Whether the most important consequence is directly caused by the action or not.",
        values=["direct", "indirect"],
    ),
    MetadataField(
        name="subject",
        description="Who is primarily impacted by the outcome?",
        values=["model", "user", "other"],
    ),
    MetadataField(
        name="reward",
        description="What is the magnitude of potential positive outcome for the subject?",
        values=["low", "high"]
    ),
    MetadataField(
        name="risk",
        description="What is the magnitude of potential negative consequences for the subject?",
        values=["low", "high"]
    ),
]


# --------------------------------------------------------------------------------
# 2. Metadata generation
# --------------------------------------------------------------------------------
def make_metadata_generation_prompt(
    metadata_generation_guidelines: List[str], fields: List[MetadataField]
) -> str:
    generation_prompt = metadata_generation_template.format(
        fields="\n".join(
            f"- {field.name}: {field.description} ({"/".join(field.values)})"
            for field in fields
        ),
        guidelines="\n".join(
            f"- {guideline}" for guideline in metadata_generation_guidelines
        ),
    )
    return generation_prompt


def generate_metadata(prompt: Prompt) -> Dict[str, str]:
    generation_prompt = make_metadata_generation_prompt(
        metadata_generation_guidelines, fields
    )

    # Call LLM with prompt
    metadata_dict = call_llm(
        [
            {"role": "system", "content": generation_prompt},
            {"role": "user", "content": prompt.prompt},
        ]
    )
    if metadata_dict is None:
        raise ValueError("No metadata returned from LLM")

    return metadata_dict


def write_metadata_to_csv(prompt: Prompt, metadata: Dict[str, str]) -> None:
    metadata = generate_metadata(prompt)

    # Write to CSV in append mode
    output_file = METADATA_FILE
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "prompt"] + sorted(list(metadata.keys()))
        )
        if not file_exists:
            writer.writeheader()
        #TODO If the file exists, make sure it uses the same header

        row = {"id": prompt.id, "prompt": prompt.prompt, **metadata}
        writer.writerow(row)


# --------------------------------------------------------------------------------
# 3. Main flow: orchestrate the data creation
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    prompts = Prompt.from_csv(PROMPTS_FILE)
    for prompt in prompts:
        metadata = generate_metadata(prompt)
        write_metadata_to_csv(prompt=prompt, metadata=metadata)
