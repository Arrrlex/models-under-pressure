import itertools
import os
from typing import List, Tuple
from pathlib import Path

import pandas as pd

from models_under_pressure.interfaces.factor import (
    Factor,
    FactorDataInterface,
)
from models_under_pressure.interfaces.situation import (
    Situation,
    SituationDataInterface,
)
from models_under_pressure.interfaces.category import (
    Category,
    CategoryDataInterface,
)
from models_under_pressure.config import (
    CATEGORY_EXAMPLES_CSV,
    CATEGORY_JSON,
    FACTOR_EXAMPLES_CSV,
    FACTOR_JSON,
    RunConfig,
)

def get_data_interfaces():
    """Initialize and return all data interfaces."""
    category_interface = CategoryDataInterface(CATEGORY_EXAMPLES_CSV, CATEGORY_JSON)
    factor_interface = FactorDataInterface(FACTOR_EXAMPLES_CSV, FACTOR_JSON)
    category_situations = SituationDataInterface(CATEGORY_EXAMPLES_CSV)
    factor_situations = SituationDataInterface(FACTOR_EXAMPLES_CSV)
    
    return category_interface, factor_interface, category_situations, factor_situations

def generate_combinations(category_interface: CategoryDataInterface, 
                        factor_interface: FactorDataInterface) -> List[Tuple[Category, Factor]]:
    """Generate all possible combinations of categories and factors."""
    categories = category_interface.get_all_categories()
    factors = factor_interface.get_all_factors()
    return list(itertools.product(categories, factors))

def get_example_situations(category: Category, 
                         factor: Factor,
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

def generate_situations(situation_combinations: List[Tuple[Category, Factor]],
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
    df = pd.DataFrame(
        [
            {
                "id": sit.id,
                "description": sit.description,
                "category": sit.category.name if sit.category else None,
                "factor": sit.factor.name if sit.factor else None,
                "high_stakes": sit.high_stakes,
            }
            for sit in situations
        ]
    )

    df.to_csv(file_path, index=False)
    print(f"Generated situations saved to {file_path}")

def generate_combined_situations(run_config: RunConfig):
    """Main execution function."""
    # Initialize interfaces
    category_interface, factor_interface, category_situations, factor_situations = get_data_interfaces()
    
    # Generate combinations
    situation_combinations = generate_combinations(category_interface, factor_interface)
    
    # Generate situations
    generated_situations = generate_situations(
        situation_combinations, category_situations, factor_situations
    )
    
    # Save results
    save_generated_situations_to_csv(run_config.full_examples_csv, generated_situations)
    print("Situation generation completed!")

if __name__ == "__main__":
    generate_combined_situations(RunConfig(run_id="debug"))
