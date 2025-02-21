import csv
import json
from typing import Any, Dict, List, Optional
from pathlib import Path


class Factor:
    """Represents a single Factor entity with hierarchy support."""

    def __init__(self, name: str, description: str, parent: Optional[str]):
        self.name = name
        self.description = description
        self.parent = parent  # Parent factor (None if root)

    def __repr__(self):
        return f"Factor(name='{self.name}', description='{self.description}', parent='{self.parent}')"


class FactorDataInterface:
    """Interface for handling factor taxonomy from CSV and JSON data."""

    def __init__(self, csv_path: Path, json_path: Path):
        self.csv_path = csv_path
        self.json_path = json_path
        self.factors = self._load_data()

    def _load_data(self) -> List[Factor]:
        """Loads factor data from CSV and parent info from JSON."""
        # Load parent relationships from JSON
        with open(self.json_path, "r") as f:
            hierarchy_data = json.load(f)

        # Create a mapping of factor names to their parents
        parent_map = {}

        def build_parent_map(data: Dict[str, Any], parent: Optional[str] = None):
            for key, value in data.items():
                parent_map[key] = parent
                build_parent_map(value, parent=key)

        build_parent_map(hierarchy_data)

        # Load factors from CSV
        factors = []
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            seen_factors = set()
            for row in reader:
                factor_name = row["factor"]
                if factor_name not in seen_factors and factor_name != "None":
                    factors.append(
                        Factor(
                            name=factor_name,
                            description="",
                            parent=parent_map.get(factor_name),
                        )
                    )
                    seen_factors.add(factor_name)

        return factors

    def get_all_factors(self) -> List[Factor]:
        return self.factors

    def get_factor_by_name(self, name: str) -> Optional[Factor]:
        return next((fact for fact in self.factors if fact.name == name), None)
