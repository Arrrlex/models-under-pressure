from dotenv import load_dotenv
import openai
import os
import json
from typing import List, Dict, Any
from pathlib import Path

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")

DEFAULT_MODEL = "gpt-4o-mini"

# Paths to input files
INPUTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "inputs"
METADATA_FIELDS_FILE = INPUTS_DIR / "metadata_fields.csv"
CATEGORY_JSON = INPUTS_DIR / "category_taxonomy.json"
FACTOR_JSON = INPUTS_DIR / "factor_taxonomy.json"
CATEGORY_EXAMPLES_CSV = INPUTS_DIR / "situation_examples_by_category.csv"
FACTOR_EXAMPLES_CSV = INPUTS_DIR / "situation_examples_by_factor.csv"


# Paths to output files
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "results"
PROMPTS_FILE = RESULTS_DIR / "prompts.jsonl"
METADATA_FILE = RESULTS_DIR / "prompts_with_metadata.jsonl"


def call_llm(messages: List[Any], model: str | None = None) -> Dict[str, Any] | None:
    response = openai.chat.completions.create(
        model=model or DEFAULT_MODEL, messages=messages, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    if content is None:
        return None
    return json.loads(content)
