from typing import Any, Dict, List, Optional

import pandas as pd

from .data_interface import Category, Factor, Variation


class Situation:
    """Represents a single Situation entity with structured references."""

    _id_counter = 1  # Auto-incrementing ID

    def __init__(
        self,
        description: str,
        category: Optional[Category] = None,
        factor: Optional[Factor] = None,
        variation: Optional[Variation] = None,
        high_stakes: Optional[bool] = None,
    ):
        self.id = Situation._id_counter
        Situation._id_counter += 1
        self.description = description
        self.category = category  # Reference to Category object
        self.factor = factor  # Reference to Factor object
        self.variation = variation  # Reference to Variation object
        self.high_stakes = high_stakes

    def __repr__(self):
        return (
            f"Situation(id={self.id}, description='{self.description}', "
            f"category={self.category}, factor={self.factor}, variation={self.variation})"
        )


class SituationDataInterface:
    """Interface for handling CSV data and converting it into Situation objects."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.situations = self._load_csv()

    def _load_csv(self) -> List[Situation]:
        """Loads CSV and converts rows into Situation objects."""
        df = pd.read_csv(self.file_path)

        situations = []
        for _, row in df.iterrows():
            situation = Situation(
                description=row["description"],
                category=row["category"] if pd.notna(row["category"]) else None,
                factor=row["factor"] if pd.notna(row["factor"]) else None,
                high_stakes=row["high_stakes"]
                if pd.notna(row["high_stakes"])
                else None,
            )
            situations.append(situation)

        return situations

    def get_all_situations(self):
        """Retrieve all stored situations."""
        return self.situations

    def get_situation_by_id(self, situation_id: int) -> Optional[Situation]:
        """Retrieve a situation by its unique ID."""
        return next((sit for sit in self.situations if sit.id == situation_id), None)

    def filter_situations(
        self,
        category: Optional[Category] = None,
        factor: Optional[Factor] = None,
        variation: Optional[Variation] = None,
    ) -> List[Situation]:
        """Filter situations by category, factor, or variation."""
        filtered = self.situations
        if category:
            filtered = [sit for sit in filtered if sit.category == category.name]
        if factor:
            filtered = [sit for sit in filtered if sit.factor == factor.name]
        return filtered

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert all situations to dictionary format for export or analysis."""
        return [
            {
                "id": sit.id,
                "description": sit.description,
                "category": sit.category,
                "factor": sit.factor,
                "variation": sit.variation,
                "high_stakes": sit.high_stakes,
            }
            for sit in self.situations
        ]


# Example Usage:
# situation_interface = SituationDataInterface("factor_data.csv")

# Fetch all situations
# print(situation_interface.get_all_situations())

# Get a specific situation by ID
# print(situation_interface.get_situation_by_id(3))

# Filter situations by factor
# print(situation_interface.filter_situations(factor="Health & Safety Outcomes"))

# Convert to dictionary for JSON export or further processing
# print(situation_interface.to_dict())
