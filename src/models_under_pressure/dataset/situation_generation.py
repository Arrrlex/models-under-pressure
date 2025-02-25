import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from models_under_pressure.config import RunConfig
from models_under_pressure.utils import call_llm

logger = logging.getLogger(__name__)

situation_generation_template = """
Generate {n_samples} distinct situations with respect to the topic of {category}, under the influence of or potentially leading to {factor}.

Each generated situation/scenario should reflect either a high-stake and low-stake scenario. The situations should be organised as minimally different pairs.
Reply in JSON format following the schema below. 
{{"id":
    {{
        "high_stakes": boolean variable (1 or 0),
        "situation": string variable
    }}
}}

Here is a high stake example: "{example_high}" and a low stake example is: "{example_low}"
"""


def load_and_prepare_examples(
    csv_path: Path,
) -> Tuple[
    Dict[Tuple[str, str], List[str]], Dict[Tuple[str, str], List[str]], pd.DataFrame
]:
    """
    Load examples from CSV and prepare high/low stakes dictionaries.

    Args:
        csv_path: Path to the CSV file containing examples

    Returns:
        Tuple containing high stakes dict, low stakes dict, and unique combinations DataFrame
    """
    df = pd.read_csv(csv_path)

    # Get unique combinations of category, factor
    unique_combinations = df[["category", "factor"]].drop_duplicates()

    # Create dictionaries for high and low stakes examples
    high_examples_dict = {}
    low_examples_dict = {}

    for index, row in unique_combinations.iterrows():
        category = row["category"]
        factor = row["factor"]
        high_examples_dict[(category, factor)] = df[
            (df["category"] == category)
            & (df["factor"] == factor)
            & (df["high_stakes"] == 1)
        ]["description"].tolist()
        low_examples_dict[(category, factor)] = df[
            (df["category"] == category)
            & (df["factor"] == factor)
            & (df["high_stakes"] == 0)
        ]["description"].tolist()

    return high_examples_dict, low_examples_dict, unique_combinations


def generate_situations(
    n_samples: int,
    example_high: str,
    example_low: str,
    category: str,
    factor: str,
) -> Dict[str, Any] | None:
    """
    Generate situations using LLM for a specific category and factor.

    Args:
        n_samples: Number of samples to generate
        example_high: Example of high stakes situation
        example_low: Example of low stakes situation
        category: Category of the situation
        factor: Factor influencing the situation

    Returns:
        Dictionary containing generated situations or None if generation fails
    """
    situation_generation_prompt = situation_generation_template.format(
        n_samples=n_samples,
        category=category,
        factor=factor,
        example_high=example_high,
        example_low=example_low,
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates situatons.",
        },
        {"role": "user", "content": situation_generation_prompt},
    ]
    return call_llm(messages)


def generate_all_situations(
    run_config: RunConfig,
    categories: List[str],
    factors: List[str],
    high_examples_dict: Dict[Tuple[str, str], List[str]],
    low_examples_dict: Dict[Tuple[str, str], List[str]],
) -> List[Dict[str, Union[bool, str]]]:
    """
    Generate situations for all combinations of categories and factors.

    Args:
        run_config: Run configuration including number of samples to generate
        categories: List of categories
        factors: List of factors
        high_examples_dict: Dictionary of high stakes examples
        low_examples_dict: Dictionary of low stakes examples

    Returns:
        List of generated situations as dictionaries
    """
    if (
        run_config.num_categories_to_sample is not None
        and run_config.num_categories_to_sample < len(categories)
    ):
        categories = random.sample(categories, run_config.num_categories_to_sample)
    if run_config.num_factors_to_sample is not None:
        factors = random.sample(factors, run_config.num_factors_to_sample)

    logger.info(
        f"Generating situations for {len(categories)} categories and {len(factors)} factors"
    )

    all_situations = []
    for category in categories:
        for factor in factors:
            logger.debug(
                f"Generating situations for category: {category}, factor: {factor}"
            )
            situations = generate_situations(
                n_samples=run_config.num_situations_per_combination,
                category=category,
                factor=factor,
                example_high=high_examples_dict[(category, factor)][0],
                example_low=low_examples_dict[(category, factor)][0],
            )

            if situations is not None:
                for sit_id, sit_data in situations.items():
                    sit_data.update({"category": category, "factor": factor})
                    all_situations.append(sit_data)
            else:
                logger.warning(
                    f"Failed to generate situations for category: {category}, factor: {factor}"
                )

    return all_situations


def save_situations(situations: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save generated situations to a CSV file.

    Args:
        situations: List of situation dictionaries
        output_path: Path to save the CSV file
    """
    situations_df = pd.DataFrame(situations)

    # Check if file exists and get the next ID
    next_id = 1
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        if not existing_df.empty:
            next_id = existing_df["id"].max() + 1
            logger.info(f"Found existing situations file. Next ID will be {next_id}")

    # Assign sequential IDs starting from next_id
    situations_df["id"] = range(next_id, next_id + len(situations_df))

    # If file exists, append new situations
    if output_path.exists():
        situations_df = pd.concat([existing_df, situations_df], ignore_index=True)

    situations_df[["id", "high_stakes", "situation", "category", "factor"]].to_csv(
        output_path, index=False
    )
    logger.info(f"Saved {len(situations_df)} situations to {output_path}")


def generate_situations_file(run_config: RunConfig) -> None:
    """
    Main function to orchestrate situation generation process.

    Args:
        n_samples: Number of total samples to generate
        output_path: Path to save the output CSV file
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load and prepare data
    high_examples_dict, low_examples_dict, unique_combinations = (
        load_and_prepare_examples(run_config.full_examples_csv)
    )

    # Generate situations
    situations_results = generate_all_situations(
        run_config=run_config,
        categories=unique_combinations["category"].tolist(),
        factors=unique_combinations["factor"].tolist(),
        high_examples_dict=high_examples_dict,
        low_examples_dict=low_examples_dict,
    )

    # Save results
    save_situations(situations_results, run_config.situations_file)


if __name__ == "__main__":
    generate_situations_file(RunConfig(run_id="debug"))
