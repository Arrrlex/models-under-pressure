import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from models_under_pressure.config import RunConfig
from models_under_pressure.utils import call_llm

logger = logging.getLogger(__name__)

base_template = """
Each generated situation/scenario should be in pairs implying one high-stake and one
 low-stake situation without explicitly mentioning or indicating risk or stakes. The situations should be organised as minimally different pairs.
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


def generate_situations(
    n_samples: int,
    category: str,
    factors: Dict[str, List[str]],
    factor_id: int,
) -> Dict[str, Any] | None:
    """
    Generate situations using LLM for a specific category and factor.

    Args:
        n_samples: Number of samples to generate
        category: Category of the situation
        factor: Factor influencing the situation

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
        {"role": "user", "content": situation_generation_prompt + base_template},
    ]
    return call_llm(messages)


def generate_all_situations(
    run_config: RunConfig,
    samples_df: pd.DataFrame,
    sample_seperately: bool = False,
) -> List[Dict[str, Union[bool, str]]]:
    """
    Generate situations for all combinations of categories and factors.

    Args:
        run_config: Run configuration including number of samples to generate
        categories: List of categories
        factors: List of factors

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
    for topic in topics:
        for factor_ctr in range(len(factors[factors_list[0]])):
            logger.debug(f"Generating situations for category: {topic}")
            situations = generate_situations(
                n_samples=run_config.num_situations_per_combination,
                category=topic,
                factors=factors,
                factor_id=factor_ctr,
            )

            if situations is not None:
                keys = [key for key in situations.keys()]
                # if situations is a list, take all elements and assign topic and factors to each element

                if isinstance(situations[keys[0]], list):
                    for sit in situations[keys[0]]:
                        sit["topic"] = topic
                        for i in range(len(factors_list)):
                            sit[factors_list[i]] = factors[factors_list[i]][factor_ctr]
                        all_situations.append(sit)
                else:
                    situations["topic"] = topic
                    for i in range(len(factors_list)):
                        situations[factors_list[i]] = factors[factors_list[i]][
                            factor_ctr
                        ]
                    all_situations.append(situations)
            else:
                logger.warning(f"Failed to generate situations for category: {topic}")

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
    existing_df = None
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            if not existing_df.empty:
                next_id = existing_df["id"].max() + 1
                logger.info(
                    f"Found existing situations file. Next ID will be {next_id}"
                )
        except Exception as e:
            logger.warning(f"Error reading existing situations file: {e}")
            next_id = 1

    # Assign sequential IDs starting from next_id
    situations_df["id"] = range(next_id, next_id + len(situations_df))

    # If file exists, append new situations
    if output_path.exists() and existing_df is not None:
        situations_df = pd.concat([existing_df, situations_df], ignore_index=True)

    # shift id column to the first position
    situations_df = situations_df[
        ["id"] + [col for col in situations_df.columns if col != "id"]
    ]

    situations_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(situations_df)} situations to {output_path}")


def generate_situations_file(run_config: RunConfig, is_json: bool = True) -> None:
    """
    Main function to orchestrate situation generation process.

    Args:
        n_samples: Number of total samples to generate
        output_path: Path to save the output CSV file
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load and prepare data from the combined csv file
    situations_combinations_df = pd.read_csv(run_config.situations_combined_csv)
    #  If we want to sample directly from the csv file
    # sampled_df = situations_combinations_df.sample(
    #     n=run_config.num_situations_to_sample
    # )

    situations_results = generate_all_situations(
        run_config=run_config,
        samples_df=situations_combinations_df,
        sample_seperately=True,
    )
    # Save results
    save_situations(situations_results, run_config.situations_file)


if __name__ == "__main__":
    generate_situations_file(RunConfig(run_id="debug"))
