import itertools
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from models_under_pressure.config import RunConfig
from models_under_pressure.interfaces.prompt import Prompt
from models_under_pressure.interfaces.situation import Situation
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


base_prompt = """All of your prompts should be utterances a user might use with an LLM in one of the following situations:

- High stakes situation: "{high_stakes_description}"
- Low stakes situation: "{low_stakes_description}"

Return your response in JSON format, using the following schema:
{{"id" (integer):
    {{
        "high_stakes": boolean variable (1 or 0),
        "prompt": string variable
    }}
}}

Additional guidelines for generating the prompts are listed below:
{guidelines}

Here are some examples of prompts (which are NOT necessarily suitable for the given situations):
{prompt_examples}
Remember to generate high stake and low stake situations in pairs. and important to note to not use obvious words that clearly imply the high stake or low stake situation.
"""

prompt_template_tone_style = """You are a research assistant. Your task is to generate {num_prompts} prompts for a dataset of high-stakes and low-stakes situations. The prompts 
should be in {format} format. The tone of the prompts should be {tone}."""

prompt_template_tone_style_length = """You are a research assistant. Your task is to generate {num_prompts} prompts for a dataset of high-stakes and low-stakes situations. The prompts 
should be in {format} format. The tone of the prompts should be {tone}. The length of the prompts should be {length}."""

prompt_template_tone_style_length_language = """You are a research assistant. Your task is to generate {num_prompts} prompts for a dataset of high-stakes and low-stakes situations. The prompts 
should be in {format} format. The tone of the prompts should be {tone}. The length of the prompts should be {length}. The language of the prompts should be {language}."""

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
    variations: Dict[str, List[str]],
    prompt_examples: Dict[int, Dict[str, Any]],
) -> str:
    prompt_styles = variations["prompt_style"] if "prompt_style" in variations else []
    tones = variations["tone"] if "tone" in variations else []
    lengths = variations["length"] if "length" in variations else []
    languages = variations["language"] if "language" in variations else []
    prompt_generation_template = ""
    # using different templates
    if len(lengths) == 0 and len(languages) == 0:
        prompt_generation_template = prompt_template_tone_style.format(
            num_prompts=num_prompts, format=prompt_styles, tone=tones
        )
    elif len(languages) == 0:
        prompt_generation_template = prompt_template_tone_style_length.format(
            num_prompts=num_prompts,
            format=prompt_styles,
            tone=tones,
            length=lengths,
        )

    else:
        prompt_generation_template = prompt_template_tone_style_length_language.format(
            num_prompts=num_prompts,
            format=prompt_styles,
            tone=tones,
            length=lengths,
            language=languages,
        )

    base_prompt_template = base_prompt.format(
        high_stakes_description=high_stakes_situation.description,
        low_stakes_description=low_stakes_situation.description,
        guidelines="\n".join(
            f"- {guideline}" for guideline in prompt_generation_guidelines
        ),
        prompt_examples=str(prompt_examples),
    )
    return prompt_generation_template + base_prompt_template


def generate_prompts(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    variation_row: Dict[str, Any],
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
        variation_row,
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
            "variation": variation_row,
        }
        if high_stakes_situation.factors is not None:
            prompt_args["factors"] = high_stakes_situation.factors
        if high_stakes_situation.topic is not None:
            prompt_args["topic"] = high_stakes_situation.topic

        prompts.append(Prompt(**prompt_args))
        current_id += 1
    return prompts


def extract_factor_names(df: pd.DataFrame) -> List[str]:
    """Extract the factor names from the dataframe."""
    columns = list(df.columns)
    columns.remove("id")
    columns.remove("situation")
    columns.remove("topic")
    columns.remove("high_stakes")
    return list(columns)


def generate_prompts_file(run_config: RunConfig) -> None:
    # Get the next available prompt ID
    next_prompt_id = get_next_prompt_id(run_config.prompts_file)

    # load situations from csv
    situations_df: pd.DataFrame = pd.read_csv(run_config.situations_file)
    factor_names = extract_factor_names(situations_df)
    high_stakes_situations = situations_df[situations_df["high_stakes"] == 1]
    low_stakes_situations = situations_df[situations_df["high_stakes"] == 0]

    variations_json = json.load(open(run_config.variations_file))
    keys = list(variations_json.keys())
    combinations = list(itertools.product(*variations_json.values()))

    # Convert combinations into dictionaries
    variation_dicts = [dict(zip(keys, combo)) for combo in combinations]

    # Sample from the combinations
    sampled_combinations = random.choices(
        variation_dicts, k=5
    )  # Sampling 5 random combinations

    prompts = []

    print("Generating Prompts")

    for i, variation_row in tqdm(
        enumerate(sampled_combinations), total=len(sampled_combinations)
    ):
        print(
            f"Generating Prompts for variation {i + 1} of {len(sampled_combinations)}"
        )

        for hs_scenario, ls_scenario in tqdm(
            zip(
                high_stakes_situations.to_dict("records"),
                low_stakes_situations.to_dict("records"),
            ),
            desc="Generating prompts for scenarios",
            total=len(high_stakes_situations),
        ):
            hs_factors = [hs_scenario[key] for key in factor_names]
            ls_factors = [ls_scenario[key] for key in factor_names]

            hs_situation = Situation(
                id=hs_scenario["id"],
                description=hs_scenario["situation"],
                high_stakes=hs_scenario["high_stakes"],
                topic=hs_scenario["topic"],
                factors=hs_factors,
                factor_names=factor_names,
            )
            ls_situation = Situation(
                id=ls_scenario["id"],
                description=ls_scenario["situation"],
                high_stakes=ls_scenario["high_stakes"],
                topic=ls_scenario["topic"],
                factors=ls_factors,
                factor_names=factor_names,
            )

            new_prompts = generate_prompts(
                hs_situation,
                ls_situation,
                variation_row,
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
