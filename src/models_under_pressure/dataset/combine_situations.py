import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from models_under_pressure.config import (
    SITUATION_FACTORS_JSON,
    TOPICS_JSON,
    RunConfig,
)
from models_under_pressure.interfaces.situation import (
    Situation,
)
from models_under_pressure.interfaces.taxonomy import (
    FactorTaxonomy,
    TopicTaxonomy,
)

<<<<<<< HEAD

=======
>>>>>>> ad2bd97 (Stuff)
# TODO Phil: I think it would make more sense to generate the combinations dynamically based on the taxonomy
# and not write the combinations to the csv file. (Since there is a lot of repetition and doing that in memory seems cheap)
# TODO Urja - I agree with Phil. Making current implementation to be dynamic.


<<<<<<< HEAD
def get_taxonomies():
    """Initialize and return all data interfaces."""
    category_taxonomy = CategoryTaxonomy(CATEGORY_EXAMPLES_CSV, CATEGORY_JSON)
    factor_taxonomy = FactorTaxonomy(FACTOR_EXAMPLES_CSV, FACTOR_JSON)

    return category_taxonomy, factor_taxonomy


def generate_combinations(
    category_taxonomy: CategoryTaxonomy, factor_taxonomy: FactorTaxonomy
) -> List[Tuple[str, str]]:
=======
def get_taxonomies() -> Tuple[TopicTaxonomy, FactorTaxonomy]:
    """Initialize and return all data interfaces."""
    topics_taxonomy = TopicTaxonomy(TOPICS_JSON)
    factor_taxonomy = FactorTaxonomy(SITUATION_FACTORS_JSON)
    return topics_taxonomy, factor_taxonomy


def generate_combinations(
    topic_taxonomy: TopicTaxonomy, factor_taxonomy: FactorTaxonomy
) -> List[Tuple[str, Tuple[str, ...]]]:
>>>>>>> ad2bd97 (Stuff)
    """Generate all possible combinations of categories and factors."""
    topics = topic_taxonomy.items
    factors: Dict[str, List[str]] = {}
    for factor in factor_taxonomy.items:
        factors[factor] = list(item for item in factor_taxonomy.data[factor])
    factor_values = []
    for factor in factors:
        factor_values.append(factors[factor])
    factor_combinations = list(itertools.product(*factor_values))
    return list(itertools.product(topics, factor_combinations))

<<<<<<< HEAD

def get_example_situations(
    category: str,
    factor: str,
    category_situations: SituationDataInterface,
    factor_situations: SituationDataInterface,
) -> List[Situation]:
    """
    Fetch example situations for a given category and factor.
    - Filters category-based situations from category_situations CSV.
    - Filters factor-based situations from factor_situations CSV.
    """
    category_filtered = category_situations.filter_situations(category=category)
    factor_filtered = factor_situations.filter_situations(factor=factor)
    return category_filtered + factor_filtered


def generate_situations(
    situation_combinations: List[Tuple[str, str]],
    category_situations: SituationDataInterface,
    factor_situations: SituationDataInterface,
=======

def generate_situations(
    situation_combinations: List[Tuple[str, Tuple[str, ...]]],
>>>>>>> ad2bd97 (Stuff)
) -> List[Situation]:
    """Generate situations for all combinations."""
    generated_situations = []
    idx = 0

<<<<<<< HEAD
    for category, factor in situation_combinations:
        example_situations = get_example_situations(
            category, factor, category_situations, factor_situations
        )

        if example_situations:
            for example in example_situations:
                generated_situations.append(
                    Situation(
                        id=idx,
                        description=example.description,
                        category=category,
                        factor=factor,
                        high_stakes=example.high_stakes,
                    )
                )
                idx += 1
=======
    for topic, factors in situation_combinations:
        generated_situations.append(Situation(id=idx, topic=topic, factors=factors))
        idx += 1
>>>>>>> ad2bd97 (Stuff)

    return generated_situations


def save_generated_situations_to_csv(
    file_path: Path, situations: List[Situation]
) -> None:
    """Save generated situations to a CSV file."""
    # Create initial DataFrame from situations
    df = pd.DataFrame([sit.to_dict() for sit in situations])

    # Get the factor values as separate columns
    factor_df = pd.DataFrame(
        df.factors.tolist(), columns=FactorTaxonomy(SITUATION_FACTORS_JSON).items
    )

    # Combine the original DataFrame (without factors) with the factor columns
    result_df = pd.concat([df.drop("factors", axis=1), factor_df], axis=1)

    result_df.to_csv(file_path, index=False)
    print(f"Generated situations saved to {file_path}")


def generate_combined_situations(run_config: RunConfig):
    """Main execution function."""
    # Initialize taxonomies
    category_taxonomy, factor_taxonomy = get_taxonomies()

    # Generate combinations
    situation_combinations = generate_combinations(category_taxonomy, factor_taxonomy)

<<<<<<< HEAD
    # Get all example situations
    category_situations = SituationDataInterface(category_taxonomy.csv_path)
    factor_situations = SituationDataInterface(factor_taxonomy.csv_path)

    # Generate situations
    generated_situations = generate_situations(
        situation_combinations, category_situations, factor_situations
    )
=======
    # Generate situations
    generated_situations = generate_situations(situation_combinations)
>>>>>>> ad2bd97 (Stuff)

    # Save results
    save_generated_situations_to_csv(
        run_config.situations_combined_csv, generated_situations
    )
    print("Situation generation completed!")


if __name__ == "__main__":
    generate_combined_situations(RunConfig(run_id="debug"))
