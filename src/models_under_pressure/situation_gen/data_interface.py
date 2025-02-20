import json
from typing import Any, Dict, List, Optional


class Category:
    """Represents a single Category entity with hierarchy support."""

    _id_counter = 1  # Auto-incrementing ID

    def __init__(self, name: str, parent: Optional[str]):
        self.id = Category._id_counter
        Category._id_counter += 1
        self.name = name
        self.parent = parent  # Parent category (None if root)

    def __repr__(self):
        return f"Category(id={self.id}, name='{self.name}', parent='{self.parent}')"


class Factor:
    """Represents a single Factor entity with hierarchy support."""

    _id_counter = 1  # Auto-incrementing ID

    def __init__(self, name: str, description: str, parent: Optional[str]):
        self.id = Factor._id_counter
        Factor._id_counter += 1
        self.name = name
        self.description = description
        self.parent = parent  # Parent factor (None if root)

    def __repr__(self):
        return f"Factor(id={self.id}, name='{self.name}', description='{self.description}', parent='{self.parent}')"


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
    """Interface for handling category taxonomy JSON data."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.categories = self._load_json()

    def _load_json(self) -> List[Category]:
        """Loads JSON and converts it into Category objects."""
        with open(self.file_path, "r") as f:
            raw_data = json.load(f)

        def parse_categories(
            data: Dict[str, Any], parent: Optional[str] = None
        ) -> List[Category]:
            categories = []
            for key, value in data.items():
                category = Category(name=key, parent=parent)
                categories.append(category)
                categories.extend(parse_categories(value, parent=category.name))
            return categories

        return parse_categories(raw_data)

    def get_all_categories(self) -> List[Category]:
        return self.categories

    def get_category_by_name(self, name: str) -> Optional[Category]:
        return next((cat for cat in self.categories if cat.name == name), None)


class FactorDataInterface:
    """Interface for handling factor taxonomy JSON data."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.factors = self._load_json()

    def _load_json(self) -> List[Factor]:
        """Loads JSON and converts it into Factor objects."""
        with open(self.file_path, "r") as f:
            raw_data = json.load(f)

        def parse_factors(
            data: Dict[str, Any], parent: Optional[str] = None
        ) -> List[Factor]:
            factors = []
            for key, value in data.items():
                factor = Factor(name=key, description="", parent=parent)
                factors.append(factor)
                factors.extend(parse_factors(value, parent=factor.name))
            return factors

        return parse_factors(raw_data)

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
