import abc
from dataclasses import dataclass
from typing import List, Dict
from models_under_pressure.dataset.prompt_generation import Prompt
from pathlib import Path
import csv
import os

from models_under_pressure.dataset.utils import call_llm, PROMPTS_FILE, METADATA_FILE, METADATA_FIELDS_FILE


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


def load_metadata_fields(file_path: Path) -> List[MetadataField]:
    fields = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fields.append(
                MetadataField(
                    name=row["name"],
                    description=row["description"],
                    values=row["values"].split("/")
                )
            )
    return fields


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


def generate_metadata(prompt: Prompt, fields: List[MetadataField]) -> Dict[str, str]:
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
    fields: List[MetadataField] = load_metadata_fields(METADATA_FIELDS_FILE)

    prompts = Prompt.from_csv(PROMPTS_FILE)
    for prompt in prompts:
        metadata = generate_metadata(prompt, fields)
        write_metadata_to_csv(prompt=prompt, metadata=metadata)

    # Now read the prompts with their metadata
    #TODO
