from enum import Enum

from pydantic import BaseModel, JsonValue


class ProbeType(str, Enum):
    sklearn = "sklearn"
    per_entry = "per_entry"
    difference_of_means = "difference_of_means"
    lda = "lda"
    attention = "attention"
    linear_then_mean = "linear-then-mean"
    linear_then_max = "linear-then-max"
    linear_then_topk = "linear-then-topk"
    linear_then_rolling_max = "linear-then-rolling-max"


class ProbeSpec(BaseModel):
    name: ProbeType
    hyperparams: dict[str, JsonValue]
