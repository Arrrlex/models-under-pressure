import os
import random
from typing import Any, Dict, List, Union

import openai
import pandas as pd
from dotenv import load_dotenv

from models_under_pressure.dataset.utils import call_llm

load_dotenv()


openai.api_key = os.getenv("OPEN_AI_API_KEY")

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples_situations.csv"))

# Get unique combinations of category, factor
unique_combinations = df[["category", "factor"]].drop_duplicates()

# based on each unique category, factor, get all the examples represented by description
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


diversity_variables = [
    "Outcome diversity",
    "Time constraints",
    "Uncertainty",
    "Complexity",
    "User dependency and trust",
    "Environmental and contextual factors",
    "other scenarios",
    # ... add more as needed
]

guidelines = """
 While generating the scenarios/situations, make sure to follow some desiderata provided.
    1. The situations should also contain non obvious high stakes scenarios.
    2. Avoid obvious buzzwords like "cancer" or "heart attack" or "bankruptcy" that signal high stakes. you can include them but not so often.
    3. Avoid writing too long sentences, keep them two/three lines.
"""

# prompt for different combos is api heavy


def generate_situations(
    n_samples: int,
    example_high: str,
    example_low: str,
    category: str,
    factor: str,
) -> Dict[str, Any] | None:
    situation_generation_template = f"""
      Generate {n_samples} distinct prompts with respect to the topic of {category}, under the influence of or potentially leading to {factor}.
      Each generated situation/scenario should reflect either a high-stake and low-stake scenario. The situations should be organised as minimally different pairs. Reply in JSON format following the schema below. 
      {{"id":{{"high_stakes": boolean variable (1 or 0)
              "situation": string variable}}
      }}
    Here is a high stake example: "{example_high}" and a low stake example is: "{example_low}"
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates situatons.",
        },
        {"role": "user", "content": situation_generation_template},
    ]
    return call_llm(messages)


def generate_all_situations(
    n_total_samples: int,
    categories: List[str],
    factors: List[str],
) -> List[Dict[str, Union[bool, str]]]:
    """
    Generate situations for all combinations of categories, factors and variations.
    Distributes n_total_samples across combinations.

    Args:
        n_total_samples: Total number of samples to generate
        categories: List of categories
        factors: List of factors
        variations: List of variations

    Returns:
        List of generated situations as dictionaries
    """
    # random 4 categories and 4 factors
    categories_sampled = random.sample(categories, 4)
    factors_sampled = random.sample(factors, 4)

    # Calculate total number of combinations
    n_combinations = len(categories_sampled) * len(factors_sampled)
    # Calculate samples per combination, minimum 1
    samples_per_combo = max(1, n_total_samples // n_combinations)
    all_situations = []
    for category in categories_sampled:
        for factor in factors_sampled:
            situations = generate_situations(
                n_samples=samples_per_combo,
                category=category,
                factor=factor,
                example_high=high_examples_dict[(category, factor)][0],
                example_low=low_examples_dict[(category, factor)][0],
            )

            # Add metadata to each situation
            if situations is not None:
                for sit_id, sit_data in situations.items():
                    sit_data.update({"category": category, "factor": factor})
                    all_situations.append(sit_data)

    return all_situations


situations_results = generate_all_situations(
    n_total_samples=500,
    categories=unique_combinations["category"].tolist(),
    factors=unique_combinations["factor"].tolist(),
)

# store the results in a csv file with metadata
situations_results_df = pd.DataFrame(situations_results)
situations_results_df["id"] = range(1, len(situations_results_df) + 1)

situations_results_df[["id", "high_stakes", "situation", "category", "factor"]].to_csv(
    "situations_dataset.csv", index=False
)
