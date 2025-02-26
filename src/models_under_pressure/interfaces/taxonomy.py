import json
from pathlib import Path


class Taxonomy:
    """Represents a taxonomy of entities with hierarchy support."""

    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.data = json.load(open(self.json_path))
        self.items = list(self.data.keys())


class TopicTaxonomy(Taxonomy):
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.data = json.load(open(self.json_path))
        self.items = self.data["topics"]


class FactorTaxonomy(Taxonomy):
    """Represents a taxonomy of factors with hierarchy support."""

    def __init__(self, json_path: Path):
        super().__init__(json_path)


if __name__ == "__main__":
    from models_under_pressure.config import (
        SITUATION_FACTORS_JSON,
        TOPICS_JSON,
    )

    category_taxonomy = TopicTaxonomy(json_path=TOPICS_JSON)

    factor_taxonomy = FactorTaxonomy(json_path=SITUATION_FACTORS_JSON)
