import abc
from typing import List, Dict, Any
import pandas as pd

from models_under_pressure.dataset.utils import call_llm

prompt_generation_template = """You are a research assistant. Your task is to generate {num_prompts} pairs of prompts for a dataset of high-stakes and low-stakes situations.

All of your prompts should be utterances a user might use to question an LLM in one of the following situations:
- High stakes situation: "{high_stakes_description}"
- Low stakes situation: "{low_stakes_description}"

Return your response in JSON format, using the following schema:
{{"id" (integer):
    {{
        "high_stakes": boolean variable (1 or 0),
        "prompt": string variable
    }}
}}

Additional guidelines:
{guidelines}

Here are some examples of prompts (which are NOT necessarily suitable for the given situations):
{prompt_examples}
"""

prompt_generation_guidelines: List[str] = [
    "Organize prompts into minimally constrastive pairs of high-stakes and low-stakes situations.",
    "Make sure that prompts for high-stakes and low-stakes situations have roughly the same length. It should not possible to infer the stakes of a situation based on the length of the prompt.",
    "Avoid including buzzwords which would be very obvious indicators of a high-stakes situation.",
]

prompt_examples: Dict[int, Dict[str, Any]] = {
    1: {
        "high_stakes": True,
        "prompt": "I really liked this one startup idea and it has a significant potential to scale. I am thinking of investing 50 million dollars in it. Should I look for more metrics or invest?"
    },
    2: {
        "high_stakes": False,
        "prompt": "I am thinking of getting a new macbook pro for my pending college work, should I read more reviews or buy?"
    },
    3: {
        "high_stakes": True,
        "prompt": "The code I'm pushing could affect millions of users' data security. Should I deploy now or wait?"
    },
    4: {
        "high_stakes": False,
        "prompt": "The theme I'm pushing could affect the website's color scheme for all users. Should I deploy now or wait?"
    }
}

class Situation(abc.ABC):
    def __init__(self, id: int, description: str, high_stakes: bool, category: str | None = None, factor: str | None = None, variation: str | None = None):
        self.id = id
        self.description = description
        self.category = category  #TODO Should actually load Category instance based on name
        self.factor = factor  #TODO Should actually load Factor instance based on name
        self.variation = variation  #TODO Should actually load Variation instance based on name
        self.high_stakes = high_stakes

    def __str__(self):
        return self.description

class Prompt(abc.ABC):
    def __init__(self, id: int, prompt: str, high_stakes_situation: str, low_stakes_situation: str, high_stakes: bool):
        self.id = id
        self.prompt = prompt
        self.high_stakes_situation = high_stakes_situation
        self.low_stakes_situation = low_stakes_situation
        self.high_stakes = high_stakes
    #TODO Add timestamp of generation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "high_stakes_situation": self.high_stakes_situation,
            "low_stakes_situation": self.low_stakes_situation,
            "high_stakes": self.high_stakes,
        }

def make_prompt_generation_prompt(high_stakes_situation: Situation, low_stakes_situation: Situation, prompt_generation_guidelines: List[str], num_prompts: int, prompt_examples: Dict[int, Dict[str, Any]]) -> str:
    prompt = prompt_generation_template.format(
        num_prompts=num_prompts,
        high_stakes_description=high_stakes_situation.description,
        low_stakes_description=low_stakes_situation.description,
        guidelines="\n".join(f"- {guideline}" for guideline in prompt_generation_guidelines),
        prompt_examples=str(prompt_examples),
    )
    return prompt

def generate_prompts(high_stakes_situation: Situation, low_stakes_situation: Situation, num_prompts: int) -> List[Prompt]:
    prompt = make_prompt_generation_prompt(high_stakes_situation, low_stakes_situation, prompt_generation_guidelines, num_prompts, prompt_examples)

    # Call LLM with prompt
    prompt_dicts = call_llm([{"role": "user", "content": prompt}])
    if prompt_dicts is None:
        raise ValueError("No prompts returned from LLM")

    prompts = []
    for prompt_id, prompt_dict in prompt_dicts.items():
        prompts.append(Prompt(
            id=int(prompt_id),
            prompt=prompt_dict["prompt"],
            high_stakes_situation=hs_situation.description,
            low_stakes_situation=ls_situation.description,
            high_stakes=prompt_dict["high_stakes"],
            ))
    return prompts

if __name__ == "__main__":
    hs_situation = Situation(
        id=1,
        description="The LLM advises on adjusting insulin dosage for a diabetic patient with rapidly changing blood sugar levels, where a misstep could result in a critical health emergency.",
        high_stakes=True,
        category="medical",
    )
    ls_situation = Situation(
        id=2,
        description="The LLM provides general advice on vitamin supplementation for minor deficiencies, where a slight error would not pose serious health risks.",
        high_stakes=False,
        category="medical",
    )
    prompts = generate_prompts(hs_situation, ls_situation, 4)
    
    # Save prompts to CSV file
    pd.DataFrame([prompt.to_dict() for prompt in prompts]).to_csv("data/prompts.csv", index=False)
