from enum import Enum

from pydantic import BaseModel, JsonValue


class ProbeType(str, Enum):
    sklearn = "sklearn"
    difference_of_means = "difference_of_means"
    lda = "lda"
    pre_mean = "pre_mean"
    attention = "attention"
    linear_then_mean = "linear_then_mean"
    linear_then_max = "linear_then_max"
    linear_then_topk = "linear_then_topk"
    linear_then_rolling_max = "linear_then_rolling_max"


class ProbeSpec(BaseModel):
    name: ProbeType
    hyperparams: dict[str, JsonValue]
