from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class Situation:
    """Represents a single Situation entity with structured references."""

    def __init__(
        self,
        id: int,
        description: Optional[str] = None,
        topic: Optional[str] = None,
        factors: Optional[Dict[str, List[str]]] = None,
        high_stakes: Optional[bool] = None,
        factor_names: Optional[List[str]] = None,
    ):
        self.id = id
        self.description = description
        self.topic = topic
        self.factors = factors
        self.high_stakes = high_stakes
        self.factor_names = factor_names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            # "description": self.description,
            "topic": self.topic,
            "factors": self.factors,
            # "high_stakes": self.high_stakes,
        }

    def __repr__(self):
        return (
            f"Situation(id={self.id}, description='{self.description}', "
            f"topic='{self.topic}', factors='{self.factors}')"
        )

    def set_factors(self, factors: Dict[str, List[str]]) -> None:
        """Set the factors for the situation."""
        if self.factor_names and self.factors:
            for factor in self.factor_names:
                self.factors[factor] = factors[factor]

    def get_all_factors(self, factor_name: str) -> List[str]:
        """Get all factors for the situation."""
        if self.factors and factor_name in self.factors:
            return self.factors[factor_name]
        return []


class SituationDataInterface:
    """Interface for handling CSV data and converting it into Situation objects."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.situations = self._load_csv()

    def _load_csv(self) -> List[Situation]:
        """Loads CSV and converts rows into Situation objects."""
        df = pd.read_csv(self.file_path)

        situations = []
        for _, row in df.iterrows():
            situation = Situation(
                id=row["id"],
                description=row["description"],
                topic=row["topic"] if pd.notna(row["topic"]) else None,
                factors=row["factors"] if pd.notna(row["factors"]) else None,
                high_stakes=row["high_stakes"],
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
        topic: Optional[str] = None,
        factors: Optional[Tuple[str, ...]] = None,
    ) -> List[Situation]:
        """todo."""
        filtered = self.situations
        if topic:
            filtered = [sit for sit in filtered if sit.topic == topic]
        if factors:
            filtered = [sit for sit in filtered if sit.factors == factors]
        return filtered

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert all situations to dictionary format for export or analysis."""
        return [
            {
                "id": sit.id,
                "description": sit.description,
                "category": sit.topic,
                "factor": sit.factors,
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
