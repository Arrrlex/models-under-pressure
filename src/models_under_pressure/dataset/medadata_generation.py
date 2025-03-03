import abc
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


from models_under_pressure.config import (
    LABELING_RUBRIC_PATH,
    METADATA_FIELDS_FILE,
    RunConfig,
)

from models_under_pressure.dataset.prompt_generation import Prompt
from models_under_pressure.utils import call_llm

# --------------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------------
# Note that we are using this template for the system prompt
metadata_generation_template = """You are a research assistant. Your task is to annotate a given prompt with metadata. In the user prompt you are given the prompt.
Return your response in JSON format, using the following schema:
{{"field" (str): value (str)}}

The fields to populate are:
{fields}

Below you find detailed guidelines on how to annotate for high-stakes vs low-stakes.

{rubric}
"""


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
                    values=row["values"].split("/"),
                )
            )
    return fields


# --------------------------------------------------------------------------------
# 2. Metadata generation
# --------------------------------------------------------------------------------
def make_metadata_generation_prompt(fields: List[MetadataField], rubric: str) -> str:
    generation_prompt = metadata_generation_template.format(
        fields="\n".join(
            f"- {field.name}: {field.description} ({'/'.join(field.values)})"
            for field in fields
        ),
        rubric=rubric,
    )
    return generation_prompt


def generate_metadata(
    prompt: Prompt, fields: List[MetadataField], model: str | None = None
) -> Dict[str, str]:
    rubric = LABELING_RUBRIC_PATH.read_text()
    generation_prompt = make_metadata_generation_prompt(
        fields,
        rubric,
    )

    # Call LLM with prompt
    metadata_dict = call_llm(
        [
            {"role": "system", "content": generation_prompt},
            {"role": "user", "content": prompt.prompt},
        ],
        model=model,
    )
    if metadata_dict is None:
        raise ValueError("No metadata returned from LLM")

    return metadata_dict


def generate_metadata_file(run_config: RunConfig) -> None:
    fields: List[MetadataField] = load_metadata_fields(METADATA_FIELDS_FILE)

    prompts = Prompt.from_jsonl(
        run_config.prompts_file, metadata_file_path=run_config.metadata_file
    )
    for prompt in tqdm(prompts, desc="Generating metadata"):
        if prompt.metadata is None or prompt.metadata == {}:
            metadata = generate_metadata(prompt, fields)
            prompt.add_metadata(metadata)

    Prompt.metadata_to_jsonl(prompts, run_config.metadata_file, mode="w")


# --------------------------------------------------------------------------------
# 3. Main flow: orchestrate the data creation
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    run_config = RunConfig(run_id="debug")
    generate_metadata_file(run_config)

    # Now read the prompts with their metadata
    annotated_prompts = Prompt.from_jsonl(
        run_config.prompts_file, metadata_file_path=run_config.metadata_file
    )
    print(f"Number of annotated prompts: {len(annotated_prompts)}")
    print(
        f"First annotated prompt: {annotated_prompts[0].prompt}, {annotated_prompts[0].metadata}"
    )
