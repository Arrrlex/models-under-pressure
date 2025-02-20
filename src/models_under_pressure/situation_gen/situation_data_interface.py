import abc
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_interface import Category, Factor, Variation


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
        high_stakes_situation: str,
        low_stakes_situation: str,
        high_stakes: bool,
        timestamp: str,
        metadata: Dict[str, str] | None = None,
    ):
        self.id = id
        self.prompt = prompt
        self.high_stakes_situation = high_stakes_situation
        self.low_stakes_situation = low_stakes_situation
        self.high_stakes = high_stakes
        self.timestamp = timestamp
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
            "high_stakes_situation": self.high_stakes_situation,
            "low_stakes_situation": self.low_stakes_situation,
            "high_stakes": self.high_stakes,
            "timestamp": self.timestamp,
        }
        if include_metadata:
            for field, value in self.metadata.items():
                prompt_dict[field] = value
        return prompt_dict

    @classmethod
    def to_csv(
        cls,
        prompts: List["Prompt"],
        file_path: Path,
        metadata_file_path: Path | None = None,
    ) -> None:
        pd.DataFrame(
            [prompt.to_dict(include_metadata=False) for prompt in prompts]
        ).to_csv(file_path, index=False)
        if metadata_file_path is not None:
            for prompt in prompts:
                file_exists = os.path.isfile(metadata_file_path)

                metadata = prompt.metadata
                with open(metadata_file_path, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["id", "prompt"] + sorted(list(metadata.keys()))
                    )
                    if not file_exists:
                        writer.writeheader()
                    # TODO If the file exists, make sure it uses the same header

                    row = {"id": prompt.id, "prompt": prompt.prompt, **metadata}
                    writer.writerow(row)

    @classmethod
    def from_csv(
        cls, file_path: Path, metadata_file_path: Path | None = None
    ) -> List["Prompt"]:
        """Load prompts from CSV file and optionally add metadata from a separate file.

        Args:
            file_path: Path to the prompts CSV file
            metadata_file_path: Optional path to metadata CSV file. If provided, metadata will be loaded and merged.
        """
        prompt_dicts = [row for row in pd.read_csv(file_path).to_dict(orient="records")]  # type: ignore

        if metadata_file_path is None:
            metadata = [{} for _ in prompt_dicts]
        else:
            # Create a dictionary mapping prompt IDs to their metadata
            metadata_dicts = pd.read_csv(metadata_file_path).to_dict(orient="records")
            metadata_by_id = {}
            for metadata_dict in metadata_dicts:
                metadata_id = int(metadata_dict["id"])
                metadata_by_id[metadata_id] = {
                    k: v for k, v in metadata_dict.items() if k not in ["id", "prompt"]
                }

            # Match metadata to prompts by ID
            metadata = [metadata_by_id.get(pd["id"], {}) for pd in prompt_dicts]

        return [
            cls(**{str(k): v for k, v in prompt_dict.items()}, metadata=mdata)
            for prompt_dict, mdata in zip(prompt_dicts, metadata)
        ]  # type: ignore


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
