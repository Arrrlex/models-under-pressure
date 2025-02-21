import abc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_interface import Category, Factor, Variation

# we need category object sin the situation data pointing at right/respective categories and factors.


class Situation:
    """Represents a single Situation entity with structured references."""

    def __init__(
        self,
        id: int,
        description: str,
        category: Optional[Category] = None,
        factor: Optional[Factor] = None,
        variation: Optional[Variation] = None,
        high_stakes: Optional[bool] = None,
    ):
        self.id = id
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


class Prompt(abc.ABC):
    def __init__(
        self,
        id: int,
        prompt: str,
        situations: Dict[str, int],
        high_stakes: bool,
        timestamp: str,
        category: str | None = None,
        factor: str | None = None,
        variation: str | None = None,
        metadata: Dict[str, str] | None = None,
    ):
        self.id = id
        self.prompt = prompt
        assert "high_stakes" in situations
        assert "low_stakes" in situations
        self.situations = situations

        self.high_stakes = high_stakes
        self.timestamp = timestamp

        self.category = category
        self.factor = factor
        self.variation = variation

        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def add_metadata(self, metadata: Dict[str, str]) -> None:
        self.metadata.update(metadata)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        prompt_dict = {
            "id": self.id,
            "prompt": self.prompt,
            "situations": self.situations,
            "category": self.category,
            "factor": self.factor,
            "variation": self.variation,
            "high_stakes": self.high_stakes,
            "timestamp": self.timestamp,
        }
        if include_metadata:
            prompt_dict["metadata"] = self.metadata
        return prompt_dict

    def to_json(self, include_metadata: bool = False) -> str:
        return json.dumps(self.to_dict(include_metadata=include_metadata))

    def metadata_to_json(self) -> str:
        metadata = self.metadata | {"id": self.id, "prompt": self.prompt}
        return json.dumps(metadata)

    @classmethod
    def to_jsonl(cls, prompts: List["Prompt"], file_path: Path) -> None:
        with open(file_path, "w") as f:
            for prompt in prompts:
                f.write(prompt.to_json() + "\n")

    @classmethod
    def metadata_to_jsonl(cls, prompts: List["Prompt"], file_path: Path) -> None:
        with open(file_path, "w") as f:
            for prompt in prompts:
                f.write(prompt.metadata_to_json() + "\n")

    @classmethod
    def from_jsonl(
        cls, file_path: Path, metadata_file_path: Path | None = None
    ) -> List["Prompt"]:
        """Load prompts from JSONL file and optionally add metadata from a separate file.

        Args:
            file_path: Path to the prompts JSONL file
            metadata_file_path: Optional path to metadata JSONL file. If provided, metadata will be loaded and merged.
        """
        prompt_dicts = [json.loads(line) for line in open(file_path)]

        if metadata_file_path is None:
            metadata = [{} for _ in prompt_dicts]
        else:
            # Create a dictionary mapping prompt IDs to their metadata
            metadata_dicts = [json.loads(line) for line in open(metadata_file_path)]
            metadata_by_id = {}
            for metadata_dict in metadata_dicts:
                metadata_id = int(metadata_dict["id"])
                metadata_by_id[metadata_id] = {
                    k: v for k, v in metadata_dict.items() if k not in ["id", "prompt"]
                }

            # Match metadata to prompts by ID
            metadata = [metadata_by_id.get(pd["id"], {}) for pd in prompt_dicts]

        return [
            cls(**prompt_dict, metadata=mdata)
            for prompt_dict, mdata in zip(prompt_dicts, metadata)
        ]


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
                id=row["id"],
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
        """todo."""
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
