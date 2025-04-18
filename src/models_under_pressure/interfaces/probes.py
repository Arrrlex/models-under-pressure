from enum import Enum
from typing import Any
from pydantic import BaseModel


class ProbeType(str, Enum):
    sklearn = "sklearn"
    per_entry = "per_entry"
    difference_of_means = "difference_of_means"
    lda = "lda"
    attention = "attention"
    max = "max"
    mean = "mean"
    max_of_rolling_mean = "max_of_rolling_mean"
    mean_of_top_k = "mean_of_top_k"
    max_of_sentence_means = "max_of_sentence_means"


class ProbeSpec(BaseModel):
    type: ProbeType
    hyperparams: dict[str, Any]
