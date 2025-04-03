from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Self, Sequence

import numpy as np
from jaxtyping import Float

from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Input,
    Label,
    LabelledDataset,
)


@dataclass
class Probe(ABC):
    model_name: str
    layer: int

    @abstractmethod
    def fit(self, dataset: LabelledDataset) -> Self: ...

    @abstractmethod
    def predict(self, dataset: BaseDataset) -> list[Label]: ...

    @abstractmethod
    def predict_proba(
        self, dataset: BaseDataset
    ) -> tuple[Activation, Float[np.ndarray, " batch_size"]]: ...

    @abstractmethod
    def per_token_predictions(
        self,
        inputs: Sequence[Input],
    ) -> Float[np.ndarray, "batch_size seq_len"]: ...


class Classifier(Protocol):
    def fit(
        self,
        X: Float[np.ndarray, "batch_size ..."],
        y: Float[np.ndarray, " batch_size"],
    ) -> Self: ...

    def predict(
        self, X: Float[np.ndarray, "batch_size ..."]
    ) -> Float[np.ndarray, " batch_size"]: ...

    def predict_proba(
        self, X: Float[np.ndarray, "batch_size ..."]
    ) -> Float[np.ndarray, "batch_size n_classes"]: ...
