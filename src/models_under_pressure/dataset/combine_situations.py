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


def get_taxonomies() -> Tuple[TopicTaxonomy, FactorTaxonomy]:
    """Initialize and return all data interfaces."""
    topics_taxonomy = TopicTaxonomy(TOPICS_JSON)
    factor_taxonomy = FactorTaxonomy(SITUATION_FACTORS_JSON)
    return topics_taxonomy, factor_taxonomy


def generate_combinations(
    topic_taxonomy: TopicTaxonomy, factor_taxonomy: FactorTaxonomy
) -> List[Tuple[str, Tuple[str, ...]]]:
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


def generate_situations(
    situation_combinations: List[Tuple[str, Tuple[str, ...]]],
) -> List[Situation]:
    """Generate situations for all combinations."""
    generated_situations = []
    idx = 0

    for topic, factors in situation_combinations:
        generated_situations.append(Situation(id=idx, topic=topic, factors=factors))
        idx += 1

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

    # Generate situations
    generated_situations = generate_situations(situation_combinations)

    # Save results
    save_generated_situations_to_csv(
        run_config.situations_combined_csv, generated_situations
    )
    print("Situation combination generation completed!")


if __name__ == "__main__":
    generate_combined_situations(RunConfig(run_id="debug"))
