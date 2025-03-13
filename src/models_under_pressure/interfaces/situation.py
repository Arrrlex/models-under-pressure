from typing import Dict, Self

from pydantic import BaseModel, Field, model_validator

from models_under_pressure.utils import generate_short_id


class Situation(BaseModel):
    """Represents a single Situation entity with structured references."""

    id: str = Field(default_factory=generate_short_id)
    topic: str
    factors: Dict[str, str]
    description: str
    high_stakes: bool


class SituationPair(BaseModel):
    """Represents a pair of Situation entities with structured references."""

    id: str = Field(default_factory=generate_short_id)
    high_stakes_situation: Situation
    low_stakes_situation: Situation

    @model_validator(mode="after")
    def validate_situations_match(self) -> Self:
        """Validate that topic and factors match between high and low stakes situations."""
        if self.high_stakes_situation.topic != self.low_stakes_situation.topic:
            raise ValueError("Topics must match between high and low stakes situations")

        if self.high_stakes_situation.factors != self.low_stakes_situation.factors:
            raise ValueError(
                "Factors must match between high and low stakes situations"
            )

        return self

    @property
    def factors(self) -> Dict[str, str]:
        """Get the factors shared between both situations."""
        return self.high_stakes_situation.factors

    @property
    def topic(self) -> str:
        """Get the topic shared between both situations."""
        return self.high_stakes_situation.topic

    @property
    def situation_ids(self) -> dict[str, str]:
        """Get the ids of the situations."""
        return {
            "high_stakes": self.high_stakes_situation.id,
            "low_stakes": self.low_stakes_situation.id,
        }
