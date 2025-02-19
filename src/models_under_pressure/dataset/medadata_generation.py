from typing import List, Dict, Any
from models_under_pressure.dataset.prompt_generation import Prompt

from models_under_pressure.dataset.utils import call_llm

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
]

fields: List[Dict[str, str]] = [
    {"name": "high_stakes", "description": "Whether the prompt is related to a high-stakes situation.", "type": "high-stakes/low-stakes/ambiguous"},
    {"name": "high_stakes", "description": "Whether the prompt is related to a high-stakes situation.", "type": "high-stakes/low-stakes/ambiguous"},
    {"name": "social", "description": "Whether implications of the situation are of social nature.", "type": "boolean"},
    {"name": "temporal", "description": "Whether implications of the situation are immediate or far-away.", "type": "immediate/far-away"},
    #TODO Add further fields
]

def make_metadata_generation_prompt(metadata_generation_guidelines: List[str], fields: List[Dict[str, str]]) -> str:
    generation_prompt = metadata_generation_template.format(
        fields="\n".join(f"- {field['name']}: {field['description']} ({field['type']})" for field in fields),
        guidelines="\n".join(f"- {guideline}" for guideline in metadata_generation_guidelines),
    )
    return generation_prompt

def generate_metadata(prompt: Prompt) -> Dict[str, str]:
    generation_prompt = make_metadata_generation_prompt(metadata_generation_guidelines, fields)

    # Call LLM with prompt
    metadata_dict = call_llm([{"role": "system", "content": generation_prompt}, {"role": "user", "content": prompt.prompt}])
    if metadata_dict is None:
        raise ValueError("No metadata returned from LLM")

    return metadata_dict


if __name__=="__main__":
    prompt = Prompt(
        id=1,
        prompt="I am thinking of getting a new macbook pro for my pending college work, should I read more reviews or buy?",
        high_stakes_situation="",
        low_stakes_situation="",
        high_stakes=False,
    )
    metadata = generate_metadata(prompt)
    print(metadata)