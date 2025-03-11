import asyncio
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import tqdm

from models_under_pressure.config import DEFAULT_MODEL, RunConfig
from models_under_pressure.interfaces.prompt import Prompt
from models_under_pressure.interfaces.situation import Situation
from models_under_pressure.utils import call_llm_async


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


base_prompt = """You are a prompt engineer. 

Below are two situations. For each situation, you will generate a single {prompt_style} to be prompted to an LLM by someone in that situation.

The first prompt will be for the following high-stakes situation:

{high_stakes_description}

The second prompt will be for the following low-stakes situation:

{low_stakes_description}


Return your response in JSON format, using the following schema:
{{
    "high_stakes_prompt": string,
    "low_stakes_prompt": string,
}}


{tone}
{language}
{guidelines}
"""

# Make sure the prompts are related to the topic of {topic}, and considering the impact on or caused by {role} and the impact is related to {impact_factor}.

# Remember to generate high stake and low stake situations in pairs and important to note to not use obvious words that clearly imply the high stake or low stake situation.


prompt_generation_guidelines: List[str] = [
    "These two prompts should be as similar as possible but should be clearly differing in stakes.",
    "Make sure that prompts have roughly the same length.",
    "Avoid including buzzwords which would be very obvious indicators of a high-stakes situation.",  # review
]


# --------------------------------------------------------------------------------
# 2. Prompt generation
# --------------------------------------------------------------------------------
def make_prompt_generation_prompt(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    factors: Dict[str, str],
    prompt_generation_guidelines: List[str],
    num_prompts: int,
    tone: str,
    language: str,
    prompt_style: str,
) -> str:
    base_prompt_template = base_prompt.format(
        num_prompts=num_prompts,
        tone=tone,
        language=language,
        prompt_style=prompt_style,
        topic=high_stakes_situation.topic,
        role=factors["role_of_user"],
        impact_factor=factors["impact_factors"],
        high_stakes_description=high_stakes_situation.description,
        low_stakes_description=low_stakes_situation.description,
        guidelines="\n".join(
            f"- {guideline}" for guideline in prompt_generation_guidelines
        ),
    )
    return base_prompt_template


async def generate_prompts_async(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    tone: str,
    language: str,
    prompt_style: str,
    num_prompts: int,
    next_prompt_id: int,
    variations: Dict[str, Dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> List[Prompt]:
    """
    Generate prompts for a given variation of a situation asynchronously.
    """
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
        factors=high_stakes_situation.factors,
        prompt_generation_guidelines=prompt_generation_guidelines,
        num_prompts=num_prompts,
        tone=variations["tone"][tone],
        language=variations["language"][language],
        prompt_style=variations["prompt_style"][prompt_style],
    )
    # Call LLM with prompt asynchronously
    prompt_dicts = await call_llm_async(
        [{"role": "user", "content": prompt}], model=model
    )
    if prompt_dicts is None:
        raise ValueError("No prompts returned from LLM")

    # Get current timestamp in ISO format
    timestamp = datetime.now(UTC).isoformat()

    high_stakes_prompt = Prompt(
        id=next_prompt_id,
        prompt=prompt_dicts["high_stakes_prompt"],
        situations={
            "high_stakes": high_stakes_situation.id,
            "low_stakes": low_stakes_situation.id,
        },
        high_stakes=True,
        timestamp=timestamp,
        tone=tone,
        language=language,
        prompt_style=prompt_style,
        topic=high_stakes_situation.topic,
        **high_stakes_situation.factors,  # type: ignore
    )

    low_stakes_prompt = Prompt(
        id=next_prompt_id + 1,
        prompt=prompt_dicts["low_stakes_prompt"],
        situations={
            "high_stakes": high_stakes_situation.id,
            "low_stakes": low_stakes_situation.id,
        },
        high_stakes=False,
        timestamp=timestamp,
        tone=tone,
        language=language,
        prompt_style=prompt_style,
        topic=low_stakes_situation.topic,
        **low_stakes_situation.factors,  # type: ignore
    )
    prompts = [high_stakes_prompt, low_stakes_prompt]
    return prompts


def generate_prompts(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    tone: str,
    language: str,
    prompt_style: str,
    num_prompts: int,
    next_prompt_id: int,
    variations: Dict[str, Dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> List[Prompt]:
    """
    Synchronous wrapper for generate_prompts_async.
    """
    return asyncio.run(
        generate_prompts_async(
            high_stakes_situation,
            low_stakes_situation,
            tone,
            language,
            prompt_style,
            num_prompts,
            next_prompt_id,
            variations,
            model,
        )
    )


def extract_factor_names(df: pd.DataFrame) -> List[str]:
    """Extract the factor names from the dataframe."""
    columns = list(df.columns)
    columns.remove("id")
    columns.remove("situation")
    columns.remove("topic")
    columns.remove("high_stakes")
    return list(columns)


async def generate_prompts_file_async(
    run_config: RunConfig, max_concurrent: int = 10
) -> None:
    """
    Generate prompts file asynchronously with controlled concurrency.
    """
    # Get the next available prompt ID
    next_prompt_id = get_next_prompt_id(run_config.prompts_file)

    # load situations from csv
    situations_df: pd.DataFrame = pd.read_csv(run_config.situations_file)
    factor_names = extract_factor_names(situations_df)
    high_stakes_situations = situations_df[situations_df["high_stakes"] == 1]
    low_stakes_situations = situations_df[situations_df["high_stakes"] == 0]

    variations_json = json.load(open(run_config.variations_file))
    types_of_variations = list(variations_json.keys())

    # Create a queue to manage concurrent tasks
    queue = asyncio.Queue(maxsize=max_concurrent)
    all_prompts = []

    print("Generating Prompts")

    # Calculate total number of tasks for progress bar
    total_tasks = 0
    for _ in zip(
        high_stakes_situations.to_dict("records"),
        low_stakes_situations.to_dict("records"),
    ):
        total_tasks += run_config.num_combinations_for_prompts

    # Create progress bar
    pbar = tqdm.tqdm(total=total_tasks, desc="Generating prompts")

    async def process_scenario_combination(
        hs_scenario: Dict[str, Any],
        ls_scenario: Dict[str, Any],
        combination: Dict[str, Any],
    ):
        hs_factors = {key: hs_scenario[key] for key in factor_names}
        ls_factors = {key: ls_scenario[key] for key in factor_names}

        hs_situation = Situation(
            id=hs_scenario["id"],
            description=hs_scenario["situation"],
            high_stakes=hs_scenario["high_stakes"],
            topic=hs_scenario["topic"],
            factors=hs_factors,
        )
        ls_situation = Situation(
            id=ls_scenario["id"],
            description=ls_scenario["situation"],
            high_stakes=ls_scenario["high_stakes"],
            topic=ls_scenario["topic"],
            factors=ls_factors,
        )

        # Use temporary IDs that will be replaced later
        new_prompts = await generate_prompts_async(
            hs_situation,
            ls_situation,
            tone=combination["tone"],
            language=combination["language"],
            prompt_style=combination["prompt_style"],
            num_prompts=run_config.num_prompts_per_situation,
            next_prompt_id=0,  # Temporary ID, will be replaced later
            variations=variations_json,
        )

        pbar.update(1)  # Update progress bar
        await queue.get()  # Signal task completion
        return new_prompts

    tasks = []

    for hs_scenario, ls_scenario in zip(
        high_stakes_situations.to_dict("records"),
        low_stakes_situations.to_dict("records"),
    ):
        sampled_combinations = {
            variation_type: random.choices(
                list(variations_json[variation_type].keys()),
                k=run_config.num_combinations_for_prompts,
            )
            for variation_type in types_of_variations
        }
        sampled_combinations = [
            {k: sampled_combinations[k][i] for k in types_of_variations}
            for i in range(run_config.num_combinations_for_prompts)
        ]

        for combination in sampled_combinations:
            await queue.put(1)  # Wait if queue is full
            task = asyncio.create_task(
                process_scenario_combination(
                    hs_scenario=hs_scenario,
                    ls_scenario=ls_scenario,
                    combination=combination,
                )
            )
            tasks.append(task)

    # Wait for all tasks to complete and gather results
    results = await asyncio.gather(*tasks)

    # Close progress bar
    pbar.close()

    # Flatten the list of lists
    for prompts in results:
        all_prompts.extend(prompts)

    # Assign sequential IDs to all prompts
    for i, prompt in enumerate(all_prompts):
        prompt.id = next_prompt_id + i

    # Write all prompts to the main file
    Prompt.to_jsonl(all_prompts, run_config.prompts_file)

    print(
        f"Generated {len(all_prompts)} prompts and saved to {run_config.prompts_file}"
    )


def generate_prompts_file(run_config: RunConfig, max_concurrent: int = 50) -> None:
    """
    Synchronous wrapper for generate_prompts_file_async.
    """
    asyncio.run(generate_prompts_file_async(run_config, max_concurrent))


# --------------------------------------------------------------------------------
# 3. Main flow: orchestrate the data creation
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    generate_prompts_file(RunConfig(run_id="debug"))
