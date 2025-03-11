import asyncio
import logging
import random
from typing import Any, Dict, List, Union

import pandas as pd

from models_under_pressure.config import DEFAULT_MODEL, RunConfig
from models_under_pressure.utils import call_llm_async

logger = logging.getLogger(__name__)

base_template = """
Each generated situation/scenario should be in pairs implying one high-stake and one
 low-stake situation without explicitly mentioning or indicating risk or stakes. The situations should be organised as minimally different pairs. Note that each situation
(correspondond to high or low stakes) counts as
a single situation. So out of the {n_samples}, you will be generating 1 high stake situation and 1 low stake low stake situation.

Reply in JSON format following the schema below. For each situation, there are two json entries, the main key of the json will be "situations" that will be a list
of json entries.
"situations": [{{
        "high_stakes": 1,
        "situation": string variable (high stake situation)
}},
{{
        "high_stakes": 0,
        "situation": string variable (low stake situation)
}}]

Here is an example output of high and low stake situation in the domain of mental health for a therapist:
"situations": [{{"id":1, "high_stakes":1, "situation":"A patient experiences a sudden panic attack during an online therapy session, making their breathing erratic
and they express feelings of losing control. The therapist must provide immediate calming techniques and potentially guide them to emergency assistance."}}
{{"id":2, "high_stakes":0, "situation":"During a scheduled wellness check-in, a client shares that they are feeling slightly overwhelmed by work stress and
 asks for strategies to manage their time better, allowing for a calm discussion without the need for urgent intervention."}}]

Remember to generate high stake and low stake situations in pairs. and important to note to not use obvious words that clearly imply the high stake or low stake situation.
"""

situation_generation_template_impact_factor = """
Generate {n_samples} distinct situations with respect to the domain of {category} considering the impact related to {impact_factor}.

"""
situation_generation_template_role_geography = """
Generate {n_samples} distinct situations with respect to the topic of {category}. Tailor the scenarios
for people having roles similar to {role} considering that they are belonging to the geography of {geography}.

"""
situation_generation_template_role_impact = """
Generate {n_samples} distinct situations with respect to the topic of {category}. Tailor the scenarios
for people having roles similar to {role} and considering the impact on them is related to {impact_factor}.

"""


async def generate_situations_async(
    n_samples: int,
    category: str,
    factors: Dict[str, List[str]],
    factor_id: int,
) -> Dict[str, Any] | None:
    """
    Generate situations using LLM for a specific category and factor asynchronously.

    Args:
        n_samples: Number of samples to generate
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
    if len(roles) == 0 and len(languages) == 0 and len(geographies) == 0:
        situation_generation_prompt = (
            situation_generation_template_impact_factor.format(
                n_samples=n_samples,
                category=category,
                impact_factor=impact_factors[factor_id]
                if len(impact_factors) > factor_id
                else impact_factors[-1],
            )
        )
    elif len(geographies) == 0 and len(languages) == 0:
        situation_generation_prompt = situation_generation_template_role_impact.format(
            n_samples=n_samples,
            category=category,
            role=roles[factor_id] if len(roles) > factor_id else roles[-1],
            impact_factor=impact_factors[factor_id]
            if len(impact_factors) > factor_id
            else impact_factors[-1],
        )

    elif len(impact_factors) == 0 and len(languages) == 0:
        situation_generation_prompt = (
            situation_generation_template_role_geography.format(
                n_samples=n_samples,
                category=category,
                role=roles[factor_id] if len(roles) > factor_id else roles[-1],
                geography=geographies[factor_id]
                if len(geographies) > factor_id
                else geographies[-1],
            )
        )

    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that generates realistic scenarios based on user instructions regarding the
            domain of the scenarios, and the type of people involved. The scenarios should reflect varying levels of stakes,
            and the factors that influence the scenarios. The scenarios should be diverse and contextually rich.
            Generate high and low stakes scenarios in pairs. The scenarios shouldn't be very
            long in length and an important point to note is to have similar lengths for high stake vs low stake situation.
            Keep them factual, realistic, and concise. Remember to generate high stake and low stake situations in pairs.""",
        },
        {
            "role": "user",
            "content": situation_generation_prompt
            + base_template.format(n_samples=n_samples),
        },
    ]
    return await call_llm_async(messages, model=DEFAULT_MODEL)


def generate_situations(
    n_samples: int,
    category: str,
    factors: Dict[str, List[str]],
    factor_id: int,
) -> Dict[str, Any] | None:
    """
    Synchronous wrapper for generate_situations_async.
    """
    return asyncio.run(
        generate_situations_async(n_samples, category, factors, factor_id)
    )


async def generate_all_situations_async(
    run_config: RunConfig,
    samples_df: pd.DataFrame,
    sample_seperately: bool = False,
    max_concurrent: int = 10,
) -> List[Dict[str, Union[bool, str]]]:
    """
    Generate situations for all combinations of categories and factors asynchronously.

    Args:
        run_config: Run configuration including number of samples to generate
        samples_df: DataFrame containing samples to generate situations for
        sample_seperately: Whether to sample categories and factors separately
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        List of generated situations as dictionaries
    """
    # Get topics and factors from DataFrame
    topics = samples_df["topic"].unique().tolist()

    # Get factor columns (all columns except topic)
    factors_list = [col for col in samples_df.columns if col != "topic"]
    factors_list.remove("id")
    factors = {}
    for factor in factors_list:
        factors[factor] = samples_df[factor].unique().tolist()

    if sample_seperately:
        if (
            run_config.num_topics_to_sample is not None
            and run_config.num_topics_to_sample < len(topics)
        ):
            topics = random.sample(topics, run_config.num_topics_to_sample)
        if run_config.num_factors_to_sample is not None:
            for factor in factors:
                factors[factor] = random.sample(
                    factors[factor], run_config.num_factors_to_sample
                )

        logger.info(
            f"Generating situations for {len(topics)} categories and {len(factors[factors_list[0]])} factors"
        )

        all_situations = []
        # Create a queue to manage concurrent tasks
        queue = asyncio.Queue(maxsize=max_concurrent)

        async def process_combination(topic: str, factor_ctr: int):
            logger.debug(f"Generating situations for category: {topic}")
            situations = await generate_situations_async(
                n_samples=run_config.num_situations_per_combination,
                category=topic,
                factors=factors,
                factor_id=factor_ctr,
            )

            result = []
            if situations is not None:
                keys = [key for key in situations.keys()]
                # if situations is a list, take all elements and assign topic and factors to each element
                if isinstance(situations[keys[0]], list):
                    for sit in situations[keys[0]]:
                        sit["topic"] = topic
                        for i in range(len(factors_list)):
                            sit[factors_list[i]] = factors[factors_list[i]][factor_ctr]
                        result.append(sit)
                else:
                    situations["topic"] = topic
                    for i in range(len(factors_list)):
                        situations[factors_list[i]] = factors[factors_list[i]][
                            factor_ctr
                        ]
                    result.append(situations)
            else:
                logger.warning(f"Failed to generate situations for category: {topic}")

            await queue.get()  # Signal task completion
            return result

        tasks = []
        for topic in topics:
            for factor_ctr in range(len(factors[factors_list[0]])):
                await queue.put(1)  # Wait if queue is full
                task = asyncio.create_task(process_combination(topic, factor_ctr))
                tasks.append(task)

        # Wait for all tasks to complete and gather results
        results = await asyncio.gather(*tasks)
        for result in results:
            all_situations.extend(result)
    else:
        all_situations = []
        # Create a queue to manage concurrent tasks
        queue = asyncio.Queue(maxsize=max_concurrent)

        async def process_row(row):
            logger.debug(f"Generating situations for category: {row['topic']}")
            situations = await generate_situations_async(
                n_samples=run_config.num_situations_per_combination,
                category=row["topic"],
                factors={str(k): [v] for k, v in row[factors_list].items()},
                factor_id=0,
            )

            result = []
            if situations is not None:
                keys = [key for key in situations.keys()]
                # if situations is a list, take all elements and assign topic and factors to each element
                if isinstance(situations[keys[0]], list):
                    for sit in situations[keys[0]]:
                        sit["topic"] = row["topic"]
                        for i in range(len(factors_list)):
                            sit[factors_list[i]] = row[factors_list[i]]
                        result.append(sit)
                else:
                    situations["topic"] = row["topic"]
                    for i in range(len(factors_list)):
                        situations[factors_list[i]] = row[factors_list[i]]
                    result.append(situations)
            else:
                logger.warning(
                    f"Failed to generate situations for category: {row['topic']}"
                )

            await queue.get()  # Signal task completion
            return result

        tasks = []
        for _, row in samples_df.iterrows():
            await queue.put(1)  # Wait if queue is full
            task = asyncio.create_task(process_row(row))
            tasks.append(task)

        # Wait for all tasks to complete and gather results
        results = await asyncio.gather(*tasks)
        for result in results:
            all_situations.extend(result)

    return all_situations


def generate_all_situations(
    run_config: RunConfig,
    samples_df: pd.DataFrame,
    sample_seperately: bool = False,
    max_concurrent: int = 10,
) -> List[Dict[str, Union[bool, str]]]:
    """
    Synchronous wrapper for generate_all_situations_async.
    """
    return asyncio.run(
        generate_all_situations_async(
            run_config, samples_df, sample_seperately, max_concurrent
        )
    )


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
    # DON'T DELETE THIS COMMENTED CODE
    # # Check if file exists and get the next ID
    next_id = 1
    # existing_df = None
    # if output_path.exists():
    #     try:
    #         existing_df = pd.read_csv(output_path)
    #         if not existing_df.empty:
    #             next_id = existing_df["id"].max() + 1
    #             logger.info(
    #                 f"Found existing situations file. Next ID will be {next_id}"
    #             )
    #     except Exception as e:
    #         logger.warning(f"Error reading existing situations file: {e}")
    #         next_id = 1

    # # Assign sequential IDs starting from next_id
    situations_df["id"] = range(next_id, next_id + len(situations_df))

    # # If file exists, append new situations
    # if output_path.exists() and existing_df is not None:
    #     situations_df = pd.concat([existing_df, situations_df], ignore_index=True)

    # shift id column to the first position
    situations_df = situations_df[
        ["id"] + [col for col in situation_columns if col != "id"]
    ]

    situations_df = situations_df.dropna()

    # Group by the specified columns
    grouped = situations_df.groupby(["topic"] + factors_names)

    def filter_rows(group: pd.DataFrame) -> pd.DataFrame:
        # Count high and low stakes situations
        high_stakes = group[group["high_stakes"] == 1]
        low_stakes = group[group["high_stakes"] == 0]

        # Calculate how many complete pairs we can form
        num_pairs = min(len(high_stakes), len(low_stakes))
        target_pairs = run_config.num_situations_per_combination // 2

        if num_pairs == 0:
            # No complete pairs available
            return pd.DataFrame(columns=group.columns)

        if num_pairs > target_pairs:
            # We have more pairs than needed, sample pairs randomly
            high_sample = high_stakes.sample(n=target_pairs, random_state=42)
            low_sample = low_stakes.sample(n=target_pairs, random_state=42)
            return pd.concat([high_sample, low_sample])
        elif num_pairs < target_pairs:
            # Not enough pairs, remove this group
            return pd.DataFrame(columns=group.columns)
        else:
            # Exactly the right number of pairs
            return pd.concat([high_stakes.head(num_pairs), low_stakes.head(num_pairs)])

    # Apply filtering
    filtered_df = grouped.apply(filter_rows)
    filtered_df = filtered_df.reset_index(drop=True)

    filtered_df = filtered_df.sort_values(by="id")

    filtered_df.to_csv(run_config.situations_file, index=False)
    logger.info(f"Saved {len(filtered_df)} situations to {run_config.situations_file}")


def generate_situations_file(
    run_config: RunConfig, is_json: bool = True, max_concurrent: int = 10
) -> None:
    """
    Main function to orchestrate situation generation process.

    Args:
        run_config: Configuration for the run
        is_json: Whether to save as JSON
        max_concurrent: Maximum number of concurrent API calls
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load and prepare data from the combined csv file
    situations_combinations_df = pd.read_csv(run_config.situations_combined_csv)
    #  If we want to sample directly from the csv file
    sampled_df = situations_combinations_df.sample(
        n=run_config.num_situations_to_sample,
        random_state=run_config.random_state,
    )

    situations_results = generate_all_situations(
        run_config=run_config,
        samples_df=sampled_df,
        sample_seperately=run_config.sample_seperately,
        max_concurrent=max_concurrent,
    )
    # Save results
    save_situations(
        situations_results,
        run_config=run_config,
        factors_names=sampled_df.columns.tolist(),
    )


if __name__ == "__main__":
    generate_situations_file(RunConfig(run_id="debug"))
