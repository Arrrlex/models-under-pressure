import asyncio
import logging
from typing import Optional

import pandas as pd

from models_under_pressure.config import RunConfig
from models_under_pressure.interfaces.situation import Situation, SituationPair
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
    Generate situations using LLM for a specific category and factor asynchronously.

    Args:
        category: Category of the situation
        factors: Factors influencing the situation
        factor_id: Index of the factor to use

    Returns:
        Dictionary containing generated situations or None if generation fails
    """

    separator = "-----"

    situation_generation_prompt = SITUATION_GENERATION_PROMPT_TEMPLATE.format(
        topic=topic,
        role=role_of_user,
        impact_factor=impact_factor,
        separator=separator,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": situation_generation_prompt},
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
        high_stakes_situation, low_stakes_situation = content.split(separator)
    except ValueError:
        logger.warning("Failed to parse situations: wrong number of parts found")
        return None

    hs = Situation(
        topic=topic,
        factors={
            "role_of_user": role_of_user,
            "impact_factor": impact_factor,
        },
        description=high_stakes_situation.strip(),
        high_stakes=True,
    )

    ls = Situation(
        topic=topic,
        factors={
            "role_of_user": role_of_user,
            "impact_factor": impact_factor,
        },
        description=low_stakes_situation.strip(),
        high_stakes=False,
    )

    return SituationPair(
        high_stakes_situation=hs,
        low_stakes_situation=ls,
    )


async def generate_situations_file(run_config: RunConfig) -> None:
    """
    Main function to orchestrate situation generation process.

    Args:
        run_config: Configuration for the run
    """
    print("Loading situations combinations from CSV...")
    situations_combinations_df = pd.read_csv(run_config.situations_combined_csv)

    logger.info(f"Sampling {run_config.num_situations_to_sample} combinations...")
    factors_combinations = situations_combinations_df.sample(
        n=run_config.num_situations_to_sample,
        random_state=run_config.random_state,
    )

    # Create a list of callables for concurrent execution
    generate_args = [
        {
            "topic": row["topic"],
            "role_of_user": row["role_of_user"],
            "impact_factor": row["impact_factors"],
            "model": run_config.model,
        }
        for _, row in factors_combinations.iterrows()
    ]

    # Call the tasks concurrently with the utility function
    situation_pairs = await async_map(
        generate_situation_pair,
        generate_args,
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
