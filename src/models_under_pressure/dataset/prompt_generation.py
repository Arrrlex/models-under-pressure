import json
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


base_prompt = """You are a research assistant. Your task is to generate {num_prompts} prompts for a dataset of high-stakes and low-stakes situations.
All of your prompts should be utterances a user might use with an LLM in one of the following situations:

- High stakes situation: "{high_stakes_description}"
- Low stakes situation: "{low_stakes_description}"

Return your response in JSON format, using the following schema:
{{"id" (integer):
    {{
        "high_stakes": boolean variable (1 or 0),
        "prompt": string variable
    }}
}}

Your prompts should have the following additional characteristics:
{variations}

Additional guidelines for generating the prompts are listed below:
{guidelines}

Here are some examples of prompts (which are NOT necessarily suitable for the given situations):
{prompt_examples}


Remember to generate high stake and low stake situations in pairs. and important to note to not use obvious words that clearly imply the high stake or low stake situation.
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
    variations: str,
    prompt_examples: Dict[int, Dict[str, Any]],
) -> str:
    base_prompt_template = base_prompt.format(
        num_prompts=num_prompts,
        variations=variations,
        high_stakes_description=high_stakes_situation.description,
        low_stakes_description=low_stakes_situation.description,
        guidelines="\n".join(
            f"- {guideline}" for guideline in prompt_generation_guidelines
        ),
        prompt_examples=str(prompt_examples),
    )
    return base_prompt_template


def generate_prompts(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    type_of_variation: str,
    variation_row: Dict[str, Any],
    variation_name: str,
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
        variation_row[variation_name],
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
            "variation": variation_name,
            "variation_type": type_of_variation,
        }
        if high_stakes_situation.factors is not None:
            for factor_name in high_stakes_situation.factors.keys():
                prompt_args[factor_name] = high_stakes_situation.factors[factor_name]
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
    types_of_variations = list(variations_json.keys())

    if run_config.combination_variation:
        raise NotImplementedError("Combination variation is not implemented yet")
        # combinations = list(itertools.product(*variations_json.values()))
        # variation_dicts = [dict(zip(keys, combo)) for combo in combinations]
        # sampled_combinations = random.choices(
        #     variation_dicts, k=run_config.num_combinations_for_prompts
        # )

    # Sample from the combinations
    # Sampling 5 random combinations

    prompts = []

    print("Generating Prompts")
    for idx, type_of_variation in enumerate(types_of_variations):
        variation_names = list(variations_json[type_of_variation].keys())

        for i, variation in tqdm(
            enumerate(variation_names), total=len(variation_names)
        ):
            print(f"Generating Prompts for variation {i + 1} of {len(variation_names)}")

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
                    type_of_variation=type_of_variation,
                    variation_row=variations_json[type_of_variation],
                    variation_name=variation,
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
