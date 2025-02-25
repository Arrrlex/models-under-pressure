from dotenv import load_dotenv
import openai
import os
import json
from typing import List, Dict, Any

from models_under_pressure.config import DEFAULT_MODEL

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")


def call_llm(messages: List[Any], model: str | None = None) -> Dict[str, Any] | None:
    response = openai.chat.completions.create(
        model=model or DEFAULT_MODEL, messages=messages, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    if content is None:
        return None
    return json.loads(content)
