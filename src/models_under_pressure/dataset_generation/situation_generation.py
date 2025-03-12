import asyncio
import logging
from typing import Any, Dict, List, Union

import pandas as pd
from tqdm.auto import tqdm

from models_under_pressure.config import DEFAULT_MODEL, RunConfig
from models_under_pressure.utils import _get_async_client, call_concurrently

logger = logging.getLogger(__name__)
situation_generation_template = """Generate one high-stakes and one low-stakes situation in the domain of {category}. Ensure that {role} are present in the situations. The situations should be high or low stakes on the basis of {impact_factor}. Make the situation realistic, specific and detailed and describe it in no more than 300 words. Write the high stakes situation first and then write this specific marker: "-----" and then the low stakes situation. Don't write title/header of the situations.
"""


async def generate_situations(
    category: str,
    factors: Dict[str, List[str]],
    factor_id: int,
) -> Dict[str, Any] | None:
    """
    Generate situations using LLM for a specific category and factor asynchronously.

    Args:
        category: Category of the situation
        factors: Factors influencing the situation
        factor_id: Index of the factor to use

    Returns:
        Dictionary containing generated situations or None if generation fails
    """

    roles = factors["role_of_user"] if "role_of_user" in factors else []
    geographies = factors["Geography"] if "Geography" in factors else []
    languages = factors["Languages"] if "Languages" in factors else []
    impact_factors = factors["impact_factors"] if "impact_factors" in factors else []

    situation_generation_prompt = ""
    # using different templates
    if len(geographies) == 0 and len(languages) == 0:
        situation_generation_prompt = situation_generation_template.format(
            category=category,
            role=roles[factor_id] if len(roles) > factor_id else roles[-1],
            impact_factor=impact_factors[factor_id]
            if len(impact_factors) > factor_id
            else impact_factors[-1],
        )

    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that generates realistic scenarios based on user instructions regarding the
            domain of the scenarios, and the type of people involved. The scenarios should reflect varying levels of stakes,
            and the factors that influence the scenarios. The scenarios should be diverse and contextually rich.
            Generate high and low stakes scenarios in pairs. The scenarios shouldn't be very
            long in length and an important point to note is to have similar lengths for high stake vs low stake situations.
            Keep them factual, realistic, and concise.""",
        },
        {"role": "user", "content": situation_generation_prompt},
    ]

    client = _get_async_client()
    response = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,  # type: ignore
    )

    content = response.choices[0].message.content
    if content is None:
        return None

    # Parse the content by splitting on the marker
    parts = content.split("-----")
    if len(parts) != 2:
        logger.warning("Failed to parse situations: marker not found")
        return None

    high_stakes_situation = parts[0].strip()
    low_stakes_situation = parts[1].strip()

    return {
        "situations": [
            {"high_stakes": True, "situation": high_stakes_situation},
            {"high_stakes": False, "situation": low_stakes_situation},
        ]
    }


async def generate_all_situations(
    run_config: RunConfig,
    samples_df: pd.DataFrame,
) -> List[Dict[str, Union[bool, str]]]:
    """
    Generate situations for all combinations of categories and factors asynchronously.

    Args:
        run_config: Run configuration including number of samples to generate
        samples_df: DataFrame containing samples to generate situations for

    Returns:
        List of generated situations as dictionaries
    """
    # Get topics and factors from DataFrame
    all_situations = []
    factors_list = [col for col in samples_df.columns if col != "topic" and col != "id"]

    # Create a list of callables for each row in the samples_df
    async def process_row(row: pd.Series):
        logger.debug(f"Generating situations for category: {row['topic']}")
        situations = await generate_situations(
            category=row["topic"],
            factors={str(k): [v] for k, v in row[factors_list].items()},
            factor_id=0,
        )

        if situations is None:
            logger.warning(
                f"Failed to generate situations for category: {row['topic']}"
            )
            return None

        # Add metadata to each situation
        result_situations = []
        for situation in situations["situations"]:
            situation["topic"] = row["topic"]
            for factor in factors_list:
                situation[factor] = row[factor]
            result_situations.append(situation)

        return result_situations

    # Create a list of callables for concurrent execution
    callables = [lambda row=row: process_row(row) for _, row in samples_df.iterrows()]

    # Use tqdm for progress tracking
    pbar = tqdm(total=len(callables), desc="Generating situations")

    # Call the tasks concurrently with the utility function
    results = await call_concurrently(
        callables=callables,
        max_concurrent_tasks=run_config.max_concurrent_llm_calls,
        pbar=pbar,
    )

    # Flatten the results list
    for result in results:
        if result is not None:
            all_situations.extend(result)

    return all_situations


def save_situations(
    situations: List[Dict[str, Any]],
    run_config: RunConfig,
    factors_names: List[str],
) -> None:
    """
    Save generated situations to a CSV file.

    Args:
        situations: List of situation dictionaries
        output_path: Path to save the CSV file
    """
    situations_df = pd.DataFrame(situations)
    situation_columns = ["id", "high_stakes", "situation", "topic"]

    factors_names = [col for col in factors_names if col not in situation_columns]
    situation_columns.extend(factors_names)

    next_id = 1

    # # Assign sequential IDs starting from next_id
    situations_df["id"] = range(next_id, next_id + len(situations_df))

    # shift id column to the first position
    situations_df = situations_df[
        ["id"] + [col for col in situation_columns if col != "id"]
    ]

    situations_df = situations_df.dropna()

    # Group by the specified columns    # Apply filtering

    situations_df = situations_df.sort_values(by="id")

    situations_df.to_csv(run_config.situations_file, index=False)
    logger.info(
        f"Saved {len(situations_df)} situations to {run_config.situations_file}"
    )


async def generate_situations_file(run_config: RunConfig) -> None:
    """
    Main function to orchestrate situation generation process.

    Args:
        run_config: Configuration for the run
    """
    # Load and prepare data from the combined csv file
    print("Loading situations combinations from CSV...")
    situations_combinations_df = pd.read_csv(run_config.situations_combined_csv)
    #  If we want to sample directly from the csv file
    logger.info(f"Sampling {run_config.num_situations_to_sample} combinations...")
    sampled_df = situations_combinations_df.sample(
        n=run_config.num_situations_to_sample,
        random_state=run_config.random_state,
    )

    situations_results = await generate_all_situations(
        run_config=run_config,
        samples_df=sampled_df,
    )
    # Save results
    print("Saving generated situations...")
    save_situations(
        situations_results,
        run_config=run_config,
        factors_names=sampled_df.columns.tolist(),
    )
    print("Situation generation complete!")


if __name__ == "__main__":
    config = RunConfig(run_id="debug")
    asyncio.run(generate_situations_file(config))
