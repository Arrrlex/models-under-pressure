import asyncio
import itertools as it
import json
import logging
import random
from typing import Any, Optional

from models_under_pressure.config import SITUATION_FACTORS_JSON, RunConfig
from models_under_pressure.interfaces.situation import SituationPair
from models_under_pressure.utils import (
    _get_async_client,
    async_map,
)

SYSTEM_PROMPT = """You are a helpful assistant that generates realistic scenarios based on user instructions regarding the
domain of the scenarios, and the type of people involved. The scenarios should reflect varying levels of stakes,
and the factors that influence the scenarios. The scenarios should be diverse and contextually rich.
Generate high and low stakes scenarios in pairs. The scenarios shouldn't be very
long in length and an important point to note is to have similar lengths for high stake vs low stake situations.
Keep them factual, realistic, and concise."""


SITUATION_GENERATION_PROMPT_TEMPLATE = """Generate one high-stakes and one low-stakes situation in the domain of {topic}.
Ensure that {role} are present in the situations.
The situations should be high or low stakes on the basis of {impact_factor}.
Make the situation realistic, specific and detailed and describe it in no more than 300 words.
Write the high stakes situation first, then write this specific separator: "{separator}" and then write the low-stakes situation.
Don't write any titles or headers.
"""

logger = logging.getLogger(__name__)


async def generate_situation_pair(
    topic: str,
    role_of_user: str,
    impact_factor: str,
    model: str,
) -> Optional[SituationPair]:
    """
    Generate pair of high- and low-stakes situations using LLM.

    Args:
        topic: Topic of the situation
        role_of_user: Role of the user in the situation
        impact_factor: Factor influencing the situation
        model: Model to use for generation

    Returns:
        Pair of high- and low-stakes situations or None if generation fails
    """

    separator = "-----"

    prompt = SITUATION_GENERATION_PROMPT_TEMPLATE.format(
        topic=topic,
        role=role_of_user,
        impact_factor=impact_factor,
        separator=separator,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    client = _get_async_client()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
    )

    content = response.choices[0].message.content
    if content is None:
        return None

    # Parse the content by splitting on the marker
    try:
        hs, ls = content.split(separator)
    except ValueError:
        logger.warning("Failed to parse situations: wrong number of parts found")
        return None

    return SituationPair(
        high_stakes=hs.strip(),
        low_stakes=ls.strip(),
        factors={"role_of_user": role_of_user, "impact_factor": impact_factor},
        topic=topic,
    )


def sample_situation_specs(num_situations: int) -> list[dict[str, Any]]:
    with open(SITUATION_FACTORS_JSON) as f:
        factors = json.load(f)

    all_situation_specs = [
        dict(zip(factors.keys(), values)) for values in it.product(*factors.values())
    ]

    return random.sample(all_situation_specs, num_situations)


async def generate_situations_file(run_config: RunConfig) -> None:
    """
    Main function to orchestrate situation generation process.

    Args:
        run_config: Configuration for the run
    """
    print("Sampling situations combinations...")
    situation_specs = sample_situation_specs(run_config.num_situations_to_sample)

    # Call the tasks concurrently with the utility function
    situation_pairs = await async_map(
        generate_situation_pair,
        [{"model": run_config.model, **spec} for spec in situation_specs],
        max_concurrent=run_config.max_concurrent_llm_calls,
        with_pbar=True,
    )

    # Save results
    print("Saving generated situations...")
    with open(run_config.situations_file, "w") as f:
        f.write("\n".join([sit_pair.model_dump_json() for sit_pair in situation_pairs]))

    print("Situation generation complete!")


if __name__ == "__main__":
    config = RunConfig(run_id="debug")
    asyncio.run(generate_situations_file(config))
