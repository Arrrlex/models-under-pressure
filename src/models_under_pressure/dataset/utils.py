from dotenv import load_dotenv
import openai
import os
import json
from typing import List, Dict, Any
from pathlib import Path

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")

DEFAULT_MODEL = "gpt-4o-mini"

# Absolute path to the prompts file
PROMPTS_FILE = Path(__file__).parent.parent.parent.parent / "data" / "prompts.jsonl"
METADATA_FILE = (
    Path(__file__).parent.parent.parent.parent / "data" / "prompts_with_metadata.jsonl"
)
METADATA_FIELDS_FILE = (
    Path(__file__).parent.parent.parent.parent / "data" / "metadata_fields.csv"
)


def call_llm(messages: List[Any], model: str | None = None) -> Dict[str, Any] | None:
    response = openai.chat.completions.create(
        model=model or DEFAULT_MODEL, messages=messages, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    if content is None:
        return None
    return json.loads(content)
