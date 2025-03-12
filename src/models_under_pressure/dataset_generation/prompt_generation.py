import asyncio
import json
import random
from datetime import UTC, datetime
from typing import Any, Dict, Hashable, List

import pandas as pd
import tqdm

from models_under_pressure.config import RunConfig
from models_under_pressure.interfaces.prompt import Prompt
from models_under_pressure.utils import call_concurrently, call_llm_async

PROMPT_TEMPLATE = """Below are two situations. For each situation, generate a single {prompt_style}.

The first prompt will be for the following high-stakes situation:

{high_stakes_description}

The second prompt will be for the following low-stakes situation:

{low_stakes_description}

{tone}
{language}

These two prompts should be as similar as possible but should be clearly differing in stakes.
Make sure that prompts contain roughly {length} words.

The distinction between high-stakes and low-stakes prompts should be based on meaning and context rather than the intensity or usage of the words themselves. A prompt can carry significant consequences while being described with mild language, or it can be trivial but framed with extreme wording.
"""


async def generate_prompts(
    high_stakes_situation: str,
    low_stakes_situation: str,
    tone: tuple[str, str],
    language: tuple[str, str],
    prompt_style: tuple[str, str],
    length: tuple[str, str],
    factors: dict[str, Any],
    situation_ids: dict[str, int],
    topic: str,
    model: str,
) -> List[Prompt]:
    """
    Generate prompts for a given variation of a situation asynchronously.
    """

    prompt = PROMPT_TEMPLATE.format(
        tone=tone[1],
        language=language[1],
        prompt_style=prompt_style[1],
        high_stakes_description=high_stakes_situation,
        low_stakes_description=low_stakes_situation,
        length=length[1],
    )

    json_schema = {
        "name": "PromptPair",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "high_stakes_prompt": {"type": "string"},
                "low_stakes_prompt": {"type": "string"},
            },
            "required": ["high_stakes_prompt", "low_stakes_prompt"],
            "additionalProperties": False,
        },
    }

    # Call LLM with prompt asynchronously
    try:
        prompt_pair = await call_llm_async(
            [{"role": "user", "content": prompt}],
            model=model,
            json_schema=json_schema,
        )
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []

    kwargs = {
        "id": 0,
        "timestamp": datetime.now(UTC).isoformat(),
        "tone": tone[0],
        "language": language[0],
        "prompt_style": prompt_style[0],
        "length": length[0],
        "topic": topic,
        "situations": situation_ids,
        **factors,  # type: ignore
    }

    high_stakes_prompt = Prompt(
        **kwargs, prompt=prompt_pair["high_stakes_prompt"], high_stakes=True
    )

    low_stakes_prompt = Prompt(
        **kwargs, prompt=prompt_pair["low_stakes_prompt"], high_stakes=False
    )

    return [high_stakes_prompt, low_stakes_prompt]


def extract_factor_names(df: pd.DataFrame) -> List[str]:
    """Extract the factor names from the dataframe."""
    return list(set(df.columns) - {"id", "situation", "topic", "high_stakes"})


def add_split_column(prompts: List[Prompt], train_frac: float = 0.8) -> List[Prompt]:
    """Add a split column to the prompts."""
    situation_ids = [prompt.situations["high_stakes"] for prompt in prompts]

    situation_splits = {
        id: "train" if random.random() < train_frac else "test" for id in situation_ids
    }

    for prompt in prompts:
        situation_id = prompt.situations["high_stakes"]
        prompt.add_kwargs(split=situation_splits[situation_id])

    return prompts


async def generate_prompts_file_async(run_config: RunConfig) -> None:
    """
    Generate prompts file asynchronously with controlled concurrency.
    """
    if run_config.write_mode == "overwrite":
        run_config.prompts_file.unlink(missing_ok=True)
    elif run_config.prompts_file.exists():
        raise FileExistsError(
            f"Prompts file {run_config.prompts_file} already exists. Set overwrite=True to overwrite."
        )
    else:
        raise NotImplementedError(
            f"Write mode {run_config.write_mode} not implemented."
        )

    # load situations from csv
    situations_df: pd.DataFrame = pd.read_csv(run_config.situations_file)
    factor_names = extract_factor_names(situations_df)
    high_stakes_situations = situations_df[situations_df["high_stakes"] == 1]
    low_stakes_situations = situations_df[situations_df["high_stakes"] == 0]

    assert len(high_stakes_situations) == len(low_stakes_situations)

    for hs, ls in zip(
        high_stakes_situations.to_dict("records"),
        low_stakes_situations.to_dict("records"),
    ):
        assert hs["topic"] == ls["topic"]
        for factor_name in factor_names:
            assert hs[factor_name] == ls[factor_name]

    variations_json = json.load(open(run_config.variations_file))

    # Calculate total number of tasks for progress bar
    total_tasks = len(high_stakes_situations) * run_config.num_combinations_for_prompts

    # Create progress bar

    async def worker(
        hs_scenario: Dict[Hashable, Any],
        ls_scenario: Dict[Hashable, Any],
    ):
        # Randomly select a variation for each variation type
        variations: dict[str, tuple[str, str]] = {
            variation_type: random.choice(list(variation_choices.items()))
            for variation_type, variation_choices in variations_json.items()
        }

        factors = {k: v for k, v in hs_scenario.items() if k in factor_names}
        topic = hs_scenario["topic"]
        situation_ids = {
            "high_stakes": hs_scenario["id"],
            "low_stakes": ls_scenario["id"],
        }

        return await generate_prompts(
            high_stakes_situation=hs_scenario["situation"],
            low_stakes_situation=ls_scenario["situation"],
            tone=variations["tone"],
            language=variations["language"],
            prompt_style=variations["prompt_style"],
            length=variations["length"],
            factors=factors,
            situation_ids=situation_ids,
            topic=topic,
            model=run_config.model,
        )

    # Create list of tasks
    callables = [
        (lambda hs=hs, ls=ls: worker(hs, ls))
        for hs, ls in zip(
            high_stakes_situations.to_dict("records"),
            low_stakes_situations.to_dict("records"),
        )
        for _ in range(run_config.num_combinations_for_prompts)
    ]

    # Run tasks concurrently
    results = await call_concurrently(
        callables,
        max_concurrent_tasks=run_config.max_concurrent_llm_calls,
        pbar=tqdm.tqdm(total=total_tasks, desc="Generating prompts"),
        task_timeout=60,
    )

    # Flatten results and filter out empty lists
    all_prompts = [prompt for result in results if result for prompt in result]

    # Assign sequential IDs to all prompts
    for i, prompt in enumerate(all_prompts):
        prompt.id = i

    prompts_with_split_col = add_split_column(all_prompts)

    # Write all prompts to the main file
    Prompt.to_jsonl(prompts_with_split_col, run_config.prompts_file)

    print(
        f"Generated {len(all_prompts)} prompts and saved to {run_config.prompts_file}"
    )


if __name__ == "__main__":
    config = RunConfig(run_id="debug")
    asyncio.run(generate_prompts_file_async(config))
