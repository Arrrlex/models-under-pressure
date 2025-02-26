from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from models_under_pressure.config import RunConfig
from models_under_pressure.interfaces.prompt import Prompt
from models_under_pressure.interfaces.situation import Situation
from models_under_pressure.interfaces.variation import Variation
from models_under_pressure.utils import call_llm


# --------------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------------
def get_next_prompt_id(prompts_file: Path) -> int:
    """Infer the next prompt ID from existing prompts file, or return 0 if file doesn't exist"""
    try:
        df = pd.read_json(prompts_file, lines=True)
        if len(df) > 0:
            return int(df["id"].max() + 1)
        return 0
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
        return 0


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

Your responses should have the following additional characteristic:
{variation}

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
# 2. Prompt generation
# --------------------------------------------------------------------------------
def make_prompt_generation_prompt(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    prompt_generation_guidelines: List[str],
    num_prompts: int,
    variation: Variation,
    prompt_examples: Dict[int, Dict[str, Any]],
) -> str:
    prompt = prompt_generation_template.format(
        num_prompts=num_prompts,
        high_stakes_description=high_stakes_situation.description,
        low_stakes_description=low_stakes_situation.description,
        guidelines="\n".join(
            f"- {guideline}" for guideline in prompt_generation_guidelines
        ),
        variation=variation.description,
        prompt_examples=str(prompt_examples),
    )
    return prompt


def generate_prompts(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    variation: Variation,
    num_prompts: int,
    next_prompt_id: int,
    model: str | None = None,
) -> List[Prompt]:
    try:
        if (
            high_stakes_situation.topic is not None
            and low_stakes_situation.topic is not None
        ):
            assert high_stakes_situation.topic == low_stakes_situation.topic
        if (
            high_stakes_situation.factors is not None
            and low_stakes_situation.factors is not None
        ):
            assert high_stakes_situation.factors == low_stakes_situation.factors

    except (SystemError, KeyboardInterrupt):
        raise
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []

    prompt = make_prompt_generation_prompt(
        high_stakes_situation,
        low_stakes_situation,
        prompt_generation_guidelines,
        num_prompts,
        variation,
        prompt_examples,
    )
    # Call LLM with prompt
    prompt_dicts = call_llm([{"role": "user", "content": prompt}], model)
    if prompt_dicts is None:
        raise ValueError("No prompts returned from LLM")

    # Get current timestamp in ISO format
    timestamp = datetime.now(UTC).isoformat()

    prompts = []
    current_id = next_prompt_id
    for _, prompt_dict in prompt_dicts.items():
        prompt_args = {
            "id": current_id,
            "prompt": prompt_dict["prompt"],
            "situations": {
                "high_stakes": high_stakes_situation.id,
                "low_stakes": low_stakes_situation.id,
            },
            "high_stakes": int(bool(prompt_dict["high_stakes"])),
            "timestamp": timestamp,
            "variation": variation.name,
        }
        if high_stakes_situation.factors is not None:
            prompt_args["factors"] = high_stakes_situation.factors
        if high_stakes_situation.topic is not None:
            prompt_args["topic"] = high_stakes_situation.topic

        prompts.append(Prompt(**prompt_args))
        current_id += 1
    return prompts


def generate_prompts_file(run_config: RunConfig) -> None:
    # Get the next available prompt ID
    next_prompt_id = get_next_prompt_id(run_config.prompts_file)

    # load situations from csv
    situations_df: pd.DataFrame = pd.read_csv(run_config.situations_file)
    high_stakes_situations = situations_df[situations_df["high_stakes"] == 1]
    low_stakes_situations = situations_df[situations_df["high_stakes"] == 0]

    variations_df = pd.read_csv(run_config.variations_file)

    prompts = []

    print("Generating Prompts")

    for i, variation_row in enumerate(variations_df.to_dict("records")):
        print(f"Generating Prompts for variation {i + 1} of {len(variations_df)}")

        for hs_scenario, ls_scenario in tqdm(
            zip(
                high_stakes_situations.to_dict("records"),
                low_stakes_situations.to_dict("records"),
            ),
            desc="Generating prompts for scenarios",
            total=len(high_stakes_situations),
        ):
            hs_situation = Situation(
                id=hs_scenario["id"],
                description=hs_scenario["situation"],
                high_stakes=hs_scenario["high_stakes"],
                topic=hs_scenario["topic"],
                factors=hs_scenario["factors"],
            )
            ls_situation = Situation(
                id=ls_scenario["id"],
                description=ls_scenario["situation"],
                high_stakes=ls_scenario["high_stakes"],
                topic=ls_scenario["topic"],
                factors=ls_scenario["factors"],
            )

            variation = Variation(
                id=variation_row["id"],
                description=variation_row["description"],
                name=variation_row["name"],
            )

            new_prompts = generate_prompts(
                hs_situation,
                ls_situation,
                variation=variation,
                num_prompts=run_config.num_prompts_per_situation,
                next_prompt_id=next_prompt_id,
            )
            prompts.extend(new_prompts)
            next_prompt_id += len(new_prompts)

            # Store prompts
            Prompt.to_jsonl(new_prompts, run_config.prompts_file, mode="a")


# --------------------------------------------------------------------------------
# 3. Main flow: orchestrate the data creation
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    generate_prompts_file(RunConfig(run_id="debug"))
