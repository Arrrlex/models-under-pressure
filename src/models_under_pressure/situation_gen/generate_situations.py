# import generate_situations.csv
# see unique combinations
# add some guidelines
#  form a template
#  ask open ai

# Read the CSV file
# mention exact path using os
import json
import os
from typing import Dict, List, Union

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPEN_AI_API_KEY")

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "generated_situations2.csv"))

# Get unique combinations of category, factor, and variation
unique_combinations = df[["category", "factor"]].drop_duplicates()

# Print the unique combinations
# based on each unique category, factor, variation, get all the examples represented by description
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
) -> Dict[str, Dict[str, Union[bool, str]]]:
    situation_generation_template = f"""
      Generate {n_samples} distinct prompts with respect to the topic of {category}, under the influence of or potentially leading to {factor}.
      Each generated situation/scenario should reflect either a high-stake and low-stake scenario. The situations should be organised as minimally different pairs. Reply in JSON format following the schema below. 
      {{"id":{{"high_stakes": boolean variable (1 or 0)
              "situation": string variable}}
      }}
    Here is a high stake example: {example_high} and a low stake example is: {example_low}
      ...
    """
    generated_situations = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates situatons.",
            },
            {"role": "user", "content": situation_generation_template},
        ],
        response_format={"type": "json_object"},
    )

    response = generated_situations.choices[0].message.content
    print(response)
    if response is None:
        return {}
    json_response = json.loads(response)
    return json_response


# wrapper function to call generate_situations for each unique combination of category, factor, variation
# if n_samples are 100, we want to distribute them throughtout the combinations equally. If combinations are too many, we can have one situation per combination.
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
    # Calculate total number of combinations
    n_combinations = len(categories) * len(factors)

    # Calculate samples per combination, minimum 1
    samples_per_combo = max(1, n_total_samples // n_combinations)
    n_ = 1
    all_situations = []
    for category in categories[:n_]:
        for factor in factors[:n_]:
            situations = generate_situations(
                n_samples=samples_per_combo,
                category=category,
                factor=factor,
                example_high=high_examples_dict[(category, factor)][0],
                example_low=low_examples_dict[(category, factor)][0],
            )

            # Add metadata to each situation
            for sit_id, sit_data in situations.items():
                sit_data.update({"category": category, "factor": factor})
                all_situations.append(sit_data)

    return all_situations


situations_results = generate_all_situations(
    n_total_samples=2,
    categories=unique_combinations.iloc[0:1]["category"].tolist(),
    factors=unique_combinations.iloc[0:1]["factor"].tolist(),
)

# store the results in a csv file with metadata
situations_results_df = pd.DataFrame(situations_results)
situations_results_df.to_csv("situations_dataset.csv", index=False)
