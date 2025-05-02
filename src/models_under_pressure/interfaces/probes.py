from enum import Enum

from pydantic import BaseModel, JsonValue


class ProbeType(str, Enum):
    sklearn = "sklearn"
    per_entry = "per_entry"
    difference_of_means = "difference_of_means"
    lda = "lda"
    per_token = "per_token"
    attention = "attention"
    simple_attention = "simple_attention"


class ProbeSpec(BaseModel):
    name: ProbeType
    hyperparams: dict[str, JsonValue]
