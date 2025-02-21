import itertools
from typing import List, Tuple
from pathlib import Path

import pandas as pd

from models_under_pressure.interfaces.taxonomy import (
    CategoryTaxonomy,
    FactorTaxonomy,
)
from models_under_pressure.interfaces.situation import (
    Situation,
    SituationDataInterface,
)
from models_under_pressure.config import (
    CATEGORY_EXAMPLES_CSV,
    CATEGORY_JSON,
    FACTOR_EXAMPLES_CSV,
    FACTOR_JSON,
    RunConfig,
)


#TODO Phil: I think it would make more sense to generate the combinations dynamically based on the taxonomy
# and not write the combinations to the csv file. (Since there is a lot of repetition and doing that in memory seems cheap)

def get_taxonomies():
    """Initialize and return all data interfaces."""
    category_taxonomy = CategoryTaxonomy(CATEGORY_EXAMPLES_CSV, CATEGORY_JSON)
    factor_taxonomy = FactorTaxonomy(FACTOR_EXAMPLES_CSV, FACTOR_JSON)
    
    return category_taxonomy, factor_taxonomy

def generate_combinations(category_taxonomy: CategoryTaxonomy, 
                        factor_taxonomy: FactorTaxonomy) -> List[Tuple[str, str]]:
    """Generate all possible combinations of categories and factors."""
    categories = category_taxonomy.nodes
    factors = factor_taxonomy.nodes
    return list(itertools.product(categories, factors))

def get_example_situations(category: str, 
                         factor: str,
                         category_situations: SituationDataInterface,
                         factor_situations: SituationDataInterface) -> List[Situation]:
    """
    Fetch example situations for a given category and factor.
    - Filters category-based situations from category_situations CSV.
    - Filters factor-based situations from factor_situations CSV.
    """
    category_filtered = category_situations.filter_situations(category=category)
    factor_filtered = factor_situations.filter_situations(factor=factor)
    return category_filtered + factor_filtered

def generate_situations(situation_combinations: List[Tuple[str, str]],
                      category_situations: SituationDataInterface,
                      factor_situations: SituationDataInterface) -> List[Situation]:
    """Generate situations for all combinations."""
    generated_situations = []
    idx = 0
    
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
    
    return generated_situations

def save_generated_situations_to_csv(file_path: Path, situations: List[Situation]) -> None:
    """Save generated situations to a CSV file."""
    df = pd.DataFrame([sit.to_dict() for sit in situations])

    df.to_csv(file_path, index=False)
    print(f"Generated situations saved to {file_path}")

def generate_combined_situations(run_config: RunConfig):
    """Main execution function."""
    # Initialize taxonomies
    category_taxonomy, factor_taxonomy = get_taxonomies()
    
    # Generate combinations
    situation_combinations = generate_combinations(category_taxonomy, factor_taxonomy)

    # Get all example situations
    category_situations = SituationDataInterface(category_taxonomy.csv_path)
    factor_situations = SituationDataInterface(factor_taxonomy.csv_path)
    
    # Generate situations
    generated_situations = generate_situations(
        situation_combinations, category_situations, factor_situations
    )
    
    # Save results
    save_generated_situations_to_csv(run_config.full_examples_csv, generated_situations)
    print("Situation generation completed!")


if __name__ == "__main__":
    generate_combined_situations(RunConfig(run_id="debug"))
