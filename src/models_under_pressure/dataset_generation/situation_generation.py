import asyncio
import logging
from typing import Any, Dict, List, Union

import pandas as pd
from tqdm.auto import tqdm

from models_under_pressure.config import DEFAULT_MODEL, RunConfig
from models_under_pressure.utils import call_llm_async

logger = logging.getLogger(__name__)

situation_generation_template = """Generate one high stake and one low stake situation in the domain of {category}. Tailor the situations
for people having roles similar to {role} and the impact of the situation should be related to {impact_factor}.

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

Please do not use obvious words or phrases that clearly imply the high stake or low stake situation. The distinction between high-stakes and low-stakes situations should be based on meaning and context rather than the intensity or usage of the words themselves."""


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
    all_situations = []
    queue = asyncio.Queue(maxsize=max_concurrent)
    pbar = tqdm(
        total=len(samples_df),
        desc="Generating situations",
    )
    factors_list = [col for col in samples_df.columns if col != "topic"]
    factors_list.remove("id")

    async def process_combination(row: pd.Series):
        logger.debug(f"Generating situations for category: {row['topic']}")
        situations = await generate_situations_async(
            n_samples=run_config.num_situations_per_combination,
            category=row["topic"],
            factors={str(k): [v] for k, v in row[factors_list].items()},
            factor_id=0,
        )
        await queue.get()  # Signal task completion
        pbar.update(1)  # Update progress bar
        return situations

    tasks = []
    for idx, row in samples_df.iterrows():
        # Create a queue to manage concurrent tasks
        # Create progress bar

        await queue.put(1)  # Wait if queue is full
        task = asyncio.create_task(process_combination(row))

        tasks.append(task)

        # Wait for all tasks to complete and gather results
    results = await asyncio.gather(*tasks)
    for result, (idx, row) in zip(results, samples_df.iterrows()):
        for situation in result["situations"]:
            situation["topic"] = row["topic"]
            situation["role_of_user"] = row["role_of_user"]
            situation["impact_factors"] = row["impact_factors"]
            all_situations.append(situation)

    pbar.close()
    return all_situations


def generate_all_situations(
    run_config: RunConfig,
    samples_df: pd.DataFrame,
    max_concurrent: int = 10,
) -> List[Dict[str, Union[bool, str]]]:
    """
    Synchronous wrapper for generate_all_situations_async.
    """
    return asyncio.run(
        generate_all_situations_async(run_config, samples_df, max_concurrent)
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

    next_id = 1

    # # Assign sequential IDs starting from next_id
    situations_df["id"] = range(next_id, next_id + len(situations_df))

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
    logger.info("Loading situations combinations from CSV...")
    situations_combinations_df = pd.read_csv(run_config.situations_combined_csv)
    #  If we want to sample directly from the csv file
    logger.info(f"Sampling {run_config.num_situations_to_sample} combinations...")
    sampled_df = situations_combinations_df.sample(
        n=run_config.num_situations_to_sample,
        random_state=run_config.random_state,
    )

    logger.info("Generating situations...")
    situations_results = generate_all_situations(
        run_config=run_config,
        samples_df=sampled_df,
        max_concurrent=max_concurrent,
    )
    # Save results
    logger.info("Saving generated situations...")
    save_situations(
        situations_results,
        run_config=run_config,
        factors_names=sampled_df.columns.tolist(),
    )
    logger.info("Situation generation complete!")


if __name__ == "__main__":
    generate_situations_file(RunConfig(run_id="debug"))
