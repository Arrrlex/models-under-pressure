from dataclasses import dataclass, field
from typing import Protocol, Self

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import Dataset, Label, LabelledDataset
from models_under_pressure.probes.model import LLMModel


class HighStakesClassifier(Protocol):
    def predict(self, dataset: Dataset) -> list[Label]: ...


class SklearnClassifier(Protocol):
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


@dataclass
class LinearProbe(HighStakesClassifier):
    _llm: LLMModel
    layer: int

    seq_pos: int | str = "all"
    _classifier: SklearnClassifier = field(
        default_factory=lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1e-3,
                random_state=42,
                fit_intercept=False,
            ),
        )
    )  # type: ignore

    def _preprocess_activations(
        self,
        activations: Activation,
    ) -> Float[np.ndarray, "batch_size embed_dim"]:
        if self.seq_pos == "all":
            acts = activations.mean_aggregation().activations
        else:
            assert isinstance(self.seq_pos, int)
            assert self.seq_pos in range(activations.activations.shape[1]), (
                f"Invalid sequence position: {self.seq_pos}"
            )
            acts = activations.activations[:, self.seq_pos, :]

        return acts

    def _fit(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
    ) -> Self:
        X = self._preprocess_activations(activations)
        X = self._preprocess_activations(activations)

        self._classifier.fit(X, y)
        return self

    def fit(self, dataset: LabelledDataset) -> Self:
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )

        print("Training probe...")
        self._fit(
            activations=activations_obj,
            y=dataset.labels_numpy(),
        )

        return self

    def predict(self, dataset: Dataset) -> list[Label]:
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )
        predictions = self._predict(activations_obj)
        return [Label.from_int(pred) for pred in predictions]

    def _predict(
        self,
        activations: Activation,
    ) -> Float[np.ndarray, " batch_size"]:
        X = self._preprocess_activations(activations)
        return self._classifier.predict(X)


def compute_accuracy(
    probe: LinearProbe,
    dataset: LabelledDataset,
    activations: Activation,
) -> float:
    pred_labels = probe._predict(activations)
    return (np.array(pred_labels) == dataset.labels_numpy()).mean()
