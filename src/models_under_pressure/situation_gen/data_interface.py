import csv
import json
from typing import Any, Dict, List, Optional


class Category:
    """Represents a single Category entity with hierarchy support."""

    def __init__(self, name: str, parent: Optional[str]):
        self.name = name
        self.parent = parent  # Parent category (None if root)

    def __repr__(self):
        return f"Category(name='{self.name}', parent='{self.parent}')"


class Factor:
    """Represents a single Factor entity with hierarchy support."""

    def __init__(self, name: str, description: str, parent: Optional[str]):
        self.name = name
        self.description = description
        self.parent = parent  # Parent factor (None if root)

    def __repr__(self):
        return f"Factor(name='{self.name}', description='{self.description}', parent='{self.parent}')"


class Variation:
    """Represents a single Variation entity."""

    _id_counter = 1  # Auto-incrementing ID

    def __init__(self, name: str, description: str):
        self.id = Variation._id_counter
        Variation._id_counter += 1
        self.name = name
        self.description = description

    def __repr__(self):
        return f"Variation(id={self.id}, name='{self.name}', description='{self.description}')"


class CategoryDataInterface:
    """Interface for handling category taxonomy from CSV and JSON data."""

    def __init__(self, csv_path: str, json_path: str):
        self.csv_path = csv_path
        self.json_path = json_path
        self.categories = self._load_data()

    def _load_data(self) -> List[Category]:
        """Loads category data from CSV and parent info from JSON."""
        # Load parent relationships from JSON
        with open(self.json_path, "r") as f:
            hierarchy_data = json.load(f)

        # Create a mapping of category names to their parents
        parent_map = {}

        def build_parent_map(data: Dict[str, Any], parent: Optional[str] = None):
            for key, value in data.items():
                parent_map[key] = parent
                build_parent_map(value, parent=key)

        build_parent_map(hierarchy_data)

        # Load categories from CSV
        categories = []
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            seen_categories = set()
            for row in reader:
                category_name = row["category"]
                if category_name not in seen_categories:
                    categories.append(
                        Category(
                            name=category_name, parent=parent_map.get(category_name)
                        )
                    )
                    seen_categories.add(category_name)

        return categories

    def get_all_categories(self) -> List[Category]:
        return self.categories

    def get_category_by_name(self, name: str) -> Optional[Category]:
        return next((cat for cat in self.categories if cat.name == name), None)


class FactorDataInterface:
    """Interface for handling factor taxonomy from CSV and JSON data."""

    def __init__(self, csv_path: str, json_path: str):
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


class VariationDataInterface:
    """Interface for handling variation types JSON data."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.variations = self._load_json()

    def _load_json(self) -> List[Variation]:
        """Loads JSON and converts it into Variation objects."""
        with open(self.file_path, "r") as f:
            raw_data = json.load(f)["variations"]

        variations = []
        for var_type, var_list in raw_data.items():
            for var in var_list:
                variations.append(Variation(name=var, description=""))

        return variations

    def get_all_variations(self) -> List[Variation]:
        return self.variations

    def get_variation_by_name(self, name: str) -> Optional[Variation]:
        return next((var for var in self.variations if var.name == name), None)
