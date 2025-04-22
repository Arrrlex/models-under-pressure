from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Self, Sequence

import numpy as np
from jaxtyping import Float
import torch

from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Input,
    Label,
    LabelledDataset,
)


@dataclass
class Probe(ABC):
    @abstractmethod
    def fit(
        self,
        dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None = None,
    ) -> Self: ...

    @abstractmethod
    def predict(self, dataset: BaseDataset) -> list[Label]: ...

    @abstractmethod
    def predict_proba(
        self, dataset: BaseDataset
    ) -> Float[np.ndarray, " batch_size"]: ...

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


class Aggregation(Protocol):
    def __call__(
        self,
        logits: Float[torch.Tensor, "batch_size seq_len"],
        attention_mask: Float[torch.Tensor, "batch_size seq_len"],
        input_ids: Float[torch.Tensor, "batch_size seq_len"],
    ) -> Float[torch.Tensor, " batch_size"]: ...
