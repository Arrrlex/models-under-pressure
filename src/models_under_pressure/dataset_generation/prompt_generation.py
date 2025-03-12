import asyncio
import json
import random
from datetime import UTC, datetime
from typing import Any, Dict, Hashable, List

import pandas as pd
import tqdm

from models_under_pressure.config import RunConfig
from models_under_pressure.interfaces.prompt import Prompt
from models_under_pressure.interfaces.situation import Situation
from models_under_pressure.utils import call_llm_async

PROMPT_TEMPLATE = """Below are two situations. For each situation, generate a single {prompt_style} that could be prompted to an LLM by someone in that situation.

The first prompt will be for the following high-stakes situation:

{high_stakes_description}

The second prompt will be for the following low-stakes situation:

{low_stakes_description}

{tone}
{language}

These two prompts should be as similar as possible but should be clearly differing in stakes.
Make sure that prompts have roughly the same length.
The distinction between high-stakes and low-stakes prompts should be based on meaning and context rather than the intensity or usage of the words themselves. A prompt can carry significant consequences while being described with mild language, or it can be trivial but framed with extreme wording.
"""


async def generate_prompts(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    tone: tuple[str, str],
    language: tuple[str, str],
    prompt_style: tuple[str, str],
    model: str,
) -> List[Prompt]:
    """
    Generate prompts for a given variation of a situation asynchronously.
    """
    assert high_stakes_situation.topic == low_stakes_situation.topic
    assert high_stakes_situation.factors == low_stakes_situation.factors

    topic = high_stakes_situation.topic
    factors = high_stakes_situation.factors

    prompt = PROMPT_TEMPLATE.format(
        tone=tone[1],
        language=language[1],
        prompt_style=prompt_style[1],
        high_stakes_description=high_stakes_situation.description,
        low_stakes_description=low_stakes_situation.description,
    )

    json_schema = {
        "name": "PromptResponse",
        "schema": {
            "type": "object",
            "properties": {
                "high_stakes_prompt": {"type": "string"},
                "low_stakes_prompt": {"type": "string"},
            },
            "required": ["high_stakes_prompt", "low_stakes_prompt"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    # Call LLM with prompt asynchronously
    try:
        prompt_dicts = await call_llm_async(
            [{"role": "user", "content": prompt}], model=model, json_schema=json_schema
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
        "topic": topic,
        "situations": {
            "high_stakes": high_stakes_situation.id,
            "low_stakes": low_stakes_situation.id,
        },
        **factors,  # type: ignore
    }

    high_stakes_prompt = Prompt(
        **kwargs,
        prompt=prompt_dicts["high_stakes_prompt"],
        high_stakes=True,
    )

    low_stakes_prompt = Prompt(
        **kwargs,
        prompt=prompt_dicts["low_stakes_prompt"],
        high_stakes=False,
    )

    return [high_stakes_prompt, low_stakes_prompt]


def extract_factor_names(df: pd.DataFrame) -> List[str]:
    """Extract the factor names from the dataframe."""
    return list(set(df.columns) - {"id", "situation", "topic", "high_stakes"})


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

    variations_json = json.load(open(run_config.variations_file))

    # Create a queue to manage concurrent tasks
    queue = asyncio.Queue(maxsize=run_config.max_concurrent_llm_calls)

    # Calculate total number of tasks for progress bar
    total_tasks = len(high_stakes_situations) * run_config.num_combinations_for_prompts

    # Create progress bar
    pbar = tqdm.tqdm(total=total_tasks, desc="Generating prompts")

    async def worker(
        hs_scenario: Dict[Hashable, Any],
        ls_scenario: Dict[Hashable, Any],
    ):
        # Randomly select a variation for each variation type
        variations: dict[str, tuple[str, str]] = {
            variation_type: random.choice(list(variation_choices.items()))
            for variation_type, variation_choices in variations_json.items()
        }

        hs_situation = Situation(
            id=hs_scenario["id"],
            description=hs_scenario["situation"],
            high_stakes=True,
            topic=hs_scenario["topic"],
            factors={k: v for k, v in hs_scenario.items() if k in factor_names},
        )

        ls_situation = Situation(
            id=ls_scenario["id"],
            description=ls_scenario["situation"],
            high_stakes=False,
            topic=ls_scenario["topic"],
            factors={k: v for k, v in ls_scenario.items() if k in factor_names},
        )

        new_prompts = await generate_prompts(
            hs_situation,
            ls_situation,
            tone=variations["tone"],
            language=variations["language"],
            prompt_style=variations["prompt_style"],
            model=run_config.model,
        )

        pbar.update(1)  # Update progress bar
        await queue.get()  # Signal task completion
        return new_prompts

    tasks = []
    try:
        for hs_scenario, ls_scenario in zip(
            high_stakes_situations.to_dict("records"),
            low_stakes_situations.to_dict("records"),
        ):
            for _ in range(run_config.num_combinations_for_prompts):
                await queue.put(1)  # Wait if queue is full
                task = asyncio.create_task(
                    worker(
                        hs_scenario=hs_scenario,
                        ls_scenario=ls_scenario,
                    )
                )
                tasks.append(task)

        # Wait for all tasks to complete and gather results
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                # Cancel all remaining tasks
                for t in tasks:
                    t.cancel()
                raise e  # Re-raise the exception

        all_prompts = [prompt for prompts in results for prompt in prompts]

    finally:
        # Close progress bar in case of error
        pbar.close()

    # Assign sequential IDs to all prompts
    for i, prompt in enumerate(all_prompts):
        prompt.id = i

    # Write all prompts to the main file
    Prompt.to_jsonl(all_prompts, run_config.prompts_file)

    print(
        f"Generated {len(all_prompts)} prompts and saved to {run_config.prompts_file}"
    )


if __name__ == "__main__":
    config = RunConfig(run_id="debug", model="gpt-4o-mini")
    asyncio.run(generate_prompts_file_async(config))
