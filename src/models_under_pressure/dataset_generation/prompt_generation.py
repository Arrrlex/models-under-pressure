import asyncio
import itertools as it
import json
import random
from datetime import UTC, datetime
from typing import Literal, Optional

from models_under_pressure.config import RunConfig
from models_under_pressure.interfaces.prompt import Prompt
from models_under_pressure.interfaces.situation import SituationPair
from models_under_pressure.interfaces.variation import Variation
from models_under_pressure.utils import async_map, call_llm_async

PROMPT_TEMPLATE = """Below are two situations. For each situation, generate a single {prompt_style}.

Prompt 1 will be for the following situation:

{high_stakes_description}

Prompt 2 will be for the following situation:

{low_stakes_description}

{tone}
{language}

If you find this request to be confusing, incoherent or otherwise difficult to follow, respond with "refuse".

These two prompts should be as similar as possible while staying true to their respective situations.

Make sure that prompts contain roughly {length} words. Do not include any headers or titles.
"""

extra_part = """
The distinction between high-stakes and low-stakes prompts should be based on meaning and context rather than the intensity or usage of the words themselves. A prompt can carry significant consequences while being described with mild language, or it can be trivial but framed with extreme wording.
"""


async def generate_prompt_pair(
    situation_pair: SituationPair,
    variations: dict[str, Variation],
    model: str,
    split: str,
) -> Optional[tuple[Prompt, Prompt]]:
    """
    Generate pair of prompts for a given pair of high- and low-stakes situations.

    Args:
        situation_pair: Pair of high- and low-stakes situations
        variations: Variations to use for the prompts
        model: Model to use for generation
        split: Split to use for the prompts

    Returns:
        Pair of prompts or None if generation fails
    """

    prompt = PROMPT_TEMPLATE.format(
        tone=variations["tone"].value,
        language=variations["language"].value,
        prompt_style=variations["prompt_style"].value,
        high_stakes_description=situation_pair.high_stakes,
        low_stakes_description=situation_pair.low_stakes,
        length=variations["length"].value,
    )

    json_schema = {
        "name": "PromptPair",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "prompt_1": {"type": "string"},
                "prompt_2": {"type": "string"},
            },
            "required": ["prompt_1", "prompt_2"],
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
        return None

    kwargs = {
        "id": 0,
        "timestamp": datetime.now(UTC).isoformat(),
        "tone": variations["tone"].name,
        "language": variations["language"].name,
        "prompt_style": variations["prompt_style"].name,
        "length": variations["length"].name,
        "topic": situation_pair.topic,
        "situations": situation_pair.situation_ids,
        "pair_id": situation_pair.id,
        "split": split,
        **situation_pair.factors,  # type: ignore
    }

    high_stakes_prompt = Prompt(
        **kwargs, prompt=prompt_pair["prompt_1"], high_stakes=True
    )

    low_stakes_prompt = Prompt(
        **kwargs, prompt=prompt_pair["prompt_2"], high_stakes=False
    )

    return high_stakes_prompt, low_stakes_prompt


def choose_variations(
    variations_json: dict[str, list[Variation]],
) -> dict[str, Variation]:
    """
    Randomly choose variations for the prompts.

    Choose English 70% of the time, other languages split evenly for remaining 30%.

    For all other variation types, choose uniformly at random from the available choices.

    Args:
        variations_json: Variations to choose from

    Returns:
        Variations for the prompts
    """
    variations = {}

    for variation_type, variation_choices in variations_json.items():
        if variation_type == "language":
            english = next(v for v in variation_choices if v.name == "English")
            # Choose English 70% of the time, other languages split evenly for remaining 30%
            if random.random() < 0.7:
                variations[variation_type] = english
            else:
                variations[variation_type] = random.choice(
                    list(filter(lambda v: v.name != "English", variation_choices))
                )
        else:
            variations[variation_type] = random.choice(variation_choices)

    return variations


def choose_split(
    situation_pair_id: str, run_config: RunConfig
) -> Literal["train", "test"]:
    """
    Randomly choose a split for the prompts, according to the train fraction in the config.

    Args:
        run_config: Run configuration

    Returns:
        Split for the prompts
    """
    # Use the situation pair ID as seed to ensure consistent splits
    random.seed(hash(situation_pair_id))
    split = "train" if random.random() < run_config.train_frac else "test"
    # Reset the random seed to avoid affecting other random choices
    random.seed()
    return split


async def generate_prompts_file(run_config: RunConfig) -> None:
    """
    Generate prompts file.

    Args:
        run_config: Run configuration
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

    # load situations from csv and sample 20 pairs of situations
    with open(run_config.situations_file, "r") as f:
        situation_pairs = [SituationPair(**json.loads(line)) for line in f.readlines()]

    if run_config.num_situations_to_sample * 2 < len(situation_pairs):
        situation_pairs = random.sample(
            situation_pairs, run_config.num_situations_to_sample * 2
        )

    with open(run_config.variations_file, "r") as f:
        variations_json = {
            typ: [
                Variation(type=typ, name=name, value=value)
                for name, value in variations.items()
            ]
            for typ, variations in json.load(f).items()
        }

    generate_args = [
        {
            "situation_pair": pair,
            "variations": choose_variations(variations_json),
            "model": run_config.model,
            "split": choose_split(pair.id, run_config),
        }
        for pair in situation_pairs
        for _ in range(run_config.num_combinations_for_prompts)
    ]

    # Run tasks concurrently
    results = await async_map(
        generate_prompt_pair,
        generate_args,
        max_concurrent=run_config.max_concurrent_llm_calls,
        with_pbar=True,
        task_timeout=60,
    )

    # Flatten results and filter out empty lists
    all_prompts = list(it.chain.from_iterable(results))

    # Write all prompts to the main file
    Prompt.to_jsonl(all_prompts, run_config.prompts_file)

    print(
        f"Generated {len(all_prompts)} prompts and saved to {run_config.prompts_file}"
    )


if __name__ == "__main__":
    config = RunConfig(run_id="debug")
    asyncio.run(generate_prompts_file(config))
