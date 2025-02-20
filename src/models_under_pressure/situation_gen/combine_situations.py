import itertools
import os
from typing import List

import pandas as pd

from models_under_pressure.situation_gen.data_interface import (
    Category,
    CategoryDataInterface,
    Factor,
    FactorDataInterface,
    VariationDataInterface,
)
from models_under_pressure.situation_gen.situation_data_interface import (
    Situation,
    SituationDataInterface,
)

# Get the directory containing this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths - using relative paths from current directory
CATEGORY_JSON = os.path.join(CURRENT_DIR, "utils", "category_taxonomy.json")
FACTOR_JSON = os.path.join(CURRENT_DIR, "utils", "factors.json")
VARIATION_JSON = os.path.join(CURRENT_DIR, "utils", "variations.json")
CATEGORY_CSV = os.path.join(CURRENT_DIR, "utils", "situation_data.csv")
FACTOR_CSV = os.path.join(CURRENT_DIR, "utils", "factor_data.csv")

# Load Interfaces
category_interface = CategoryDataInterface(CATEGORY_JSON)
factor_interface = FactorDataInterface(FACTOR_JSON)
variation_interface = VariationDataInterface(VARIATION_JSON)
category_situations = SituationDataInterface(CATEGORY_CSV)
factor_situations = SituationDataInterface(FACTOR_CSV)

# Step 1: Generate all category, factor, and variation combinations
categories = category_interface.get_all_categories()
factors = factor_interface.get_all_factors()
variations = variation_interface.get_all_variations()

situation_combinations = list(itertools.product(categories, factors, variations))


# Step 2: Load Example Situations from CSVs
def get_example_situations(category: Category, factor: Factor) -> List[Situation]:
    """
    Fetch example situations for a given category and factor.
    - Filters category-based situations from category_situations CSV.
    - Filters factor-based situations from factor_situations CSV.
    """
    category_filtered = category_situations.filter_situations(category=category)
    factor_filtered = factor_situations.filter_situations(factor=factor)

    return category_filtered + factor_filtered  # Combine results


# Step 3: Generate Situations for Each Combination
generated_situations = []
# for now variation is not used
for category, factor, variation in situation_combinations:
    example_situations = get_example_situations(category, factor)

    # Create new situations based on the combinations and example situations
    if example_situations:
        for example in example_situations:
            generated_situations.append(
                Situation(
                    description=example.description,
                    category=category,
                    factor=factor,
                    variation=variation,
                    high_stakes=example.high_stakes,
                )
            )


# Step 4: Save Generated Situations to CSV
def save_generated_situations_to_csv(file_path: str, situations: List[Situation]):
    """Save generated situations to a CSV file."""
    df = pd.DataFrame(
        [
            {
                "id": sit.id,
                "description": sit.description,
                "category": sit.category.name if sit.category else None,
                "factor": sit.factor.name if sit.factor else None,
                "variation": sit.variation.name if sit.variation else None,
                "high_stakes": sit.high_stakes,
            }
            for sit in situations
        ]
    )

    df.to_csv(file_path, index=False)
    print(f"Generated situations saved to {file_path}")


# Update OUTPUT_CSV to use current directory
OUTPUT_CSV = os.path.join(CURRENT_DIR, "generated_situations2.csv")
save_generated_situations_to_csv(OUTPUT_CSV, generated_situations)

print("Situation generation completed!")
