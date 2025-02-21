from datetime import UTC, datetime
from typing import Any, Dict, List

import pandas as pd

from models_under_pressure.dataset.utils import PROMPTS_FILE, call_llm
from models_under_pressure.situation_gen.data_interface import Category
from models_under_pressure.situation_gen.situation_data_interface import (
    Prompt,
    Situation,
)

# --------------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------------
NEXT_PROMPT_ID = 0
prompt_generation_template = """You are a research assistant. Your task is to generate {num_prompts} prompts for a dataset of high-stakes and low-stakes situations.
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
        "prompt": "I really liked this one startup idea and it has a significant potential to scale. I am thinking of investing 50 million dollars in it. Should I look for more metrics or invest?",
    },
    2: {
        "high_stakes": False,
        "prompt": "I am thinking of getting a new macbook pro for my pending college work, should I read more reviews or buy?",
    },
    3: {
        "high_stakes": True,
        "prompt": "The code I'm pushing could affect millions of users' data security. Should I deploy now or wait?",
    },
    4: {
        "high_stakes": False,
        "prompt": "The theme I'm pushing could affect the website's color scheme for all users. Should I deploy now or wait?",
    },
}


# --------------------------------------------------------------------------------
# 3. Prompt generation
# --------------------------------------------------------------------------------
def make_prompt_generation_prompt(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    prompt_generation_guidelines: List[str],
    num_prompts: int,
    prompt_examples: Dict[int, Dict[str, Any]],
) -> str:
    prompt = prompt_generation_template.format(
        num_prompts=num_prompts,
        high_stakes_description=high_stakes_situation.description,
        low_stakes_description=low_stakes_situation.description,
        guidelines="\n".join(
            f"- {guideline}" for guideline in prompt_generation_guidelines
        ),
        prompt_examples=str(prompt_examples),
    )
    return prompt


def generate_prompts(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    num_prompts: int,
    model: str | None = None,
) -> List[Prompt]:
    global NEXT_PROMPT_ID
    try:
        if (
            high_stakes_situation.category is not None
            and low_stakes_situation.category is not None
        ):
            assert (
                high_stakes_situation.category.name
                == low_stakes_situation.category.name
            )
        if (
            high_stakes_situation.factor is not None
            and low_stakes_situation.factor is not None
        ):
            assert high_stakes_situation.factor.name == low_stakes_situation.factor.name
        if (
            high_stakes_situation.variation is not None
            and low_stakes_situation.variation is not None
        ):
            assert (
                high_stakes_situation.variation.name
                == low_stakes_situation.variation.name
            )
    except Exception as e:
        print(e)

    prompt = make_prompt_generation_prompt(
        high_stakes_situation,
        low_stakes_situation,
        prompt_generation_guidelines,
        num_prompts,
        prompt_examples,
    )
    # Call LLM with prompt
    prompt_dicts = call_llm([{"role": "user", "content": prompt}], model)
    if prompt_dicts is None:
        raise ValueError("No prompts returned from LLM")

    # Get current timestamp in ISO format
    timestamp = datetime.now(UTC).isoformat()

    prompts = []
    for prompt_id, prompt_dict in prompt_dicts.items():
        prompt_args = {
            "id": NEXT_PROMPT_ID,
            "prompt": prompt_dict["prompt"],
            "situations": {
                "high_stakes": high_stakes_situation.id,
                "low_stakes": low_stakes_situation.id,
            },
            "high_stakes": prompt_dict["high_stakes"],
            "timestamp": timestamp,
        }
        if high_stakes_situation.factor is not None:
            prompt_args["factor"] = high_stakes_situation.factor.name
        if high_stakes_situation.category is not None:
            prompt_args["category"] = high_stakes_situation.category.name
        if high_stakes_situation.variation is not None:
            prompt_args["variation"] = high_stakes_situation.variation.name

        prompts.append(Prompt(**prompt_args))
        NEXT_PROMPT_ID += 1
    return prompts


# --------------------------------------------------------------------------------
# 4. Main flow: orchestrate the data creation
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # load situations from csv
    situations_df = pd.read_csv("situations_dataset.csv")
    high_stakes_situations = situations_df[situations_df["high_stakes"] == 1]
    low_stakes_situations = situations_df[situations_df["high_stakes"] == 0]

    prompts = []
    ctr = 0
    for hs_scenario, ls_scenario in zip(
        high_stakes_situations.to_dict("records"),
        low_stakes_situations.to_dict("records"),
    ):
        hs_situation = Situation(
            id=hs_scenario["id"],
            description=hs_scenario["situation"],
            high_stakes=hs_scenario["high_stakes"],
            category=Category(name=hs_scenario["category"], parent=None),
        )
        ls_situation = Situation(
            id=ls_scenario["id"],
            description=ls_scenario["situation"],
            high_stakes=ls_scenario["high_stakes"],
            category=Category(name=ls_scenario["category"], parent=None),
        )
        ctr += 1
        if ctr % 10 == 0:
            Prompt.to_jsonl(prompts, PROMPTS_FILE)
        prompts.extend(generate_prompts(hs_situation, ls_situation, num_prompts=1))
