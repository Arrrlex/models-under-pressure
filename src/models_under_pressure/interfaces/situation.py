from typing import Dict

from pydantic import BaseModel, Field

from models_under_pressure.utils import generate_short_id


class Situation(BaseModel):
    """Represents a single Situation entity with structured references."""

    id: str = Field(default_factory=generate_short_id)
    topic: str
    factors: Dict[str, str]
    description: str
    high_stakes: bool


class SituationPair(BaseModel):
    high_stakes: str
    low_stakes: str
    factors: Dict[str, str]
    topic: str

    id: str = Field(default_factory=generate_short_id)
    high_stakes_id: str = Field(default_factory=generate_short_id)
    low_stakes_id: str = Field(default_factory=generate_short_id)

    @property
    def situation_ids(self) -> dict[str, str]:
        return {
            "high_stakes": self.high_stakes_id,
            "low_stakes": self.low_stakes_id,
        }

    @property
    def high_stakes_situation(self) -> Situation:
        return Situation(
            id=self.high_stakes_id,
            topic=self.topic,
            factors=self.factors,
            description=self.high_stakes,
            high_stakes=True,
        )

    @property
    def low_stakes_situation(self) -> Situation:
        return Situation(
            id=self.low_stakes_id,
            topic=self.topic,
            factors=self.factors,
            description=self.low_stakes,
            high_stakes=False,
        )
