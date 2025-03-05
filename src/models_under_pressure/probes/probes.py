from dataclasses import dataclass, field
from typing import Protocol, Self

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
        activations: Float[np.ndarray, "batch_size seq_len embed_dim"],
        attention_mask: Float[np.ndarray, "batch_size seq_len"],
    ) -> Float[np.ndarray, "batch_size embed_dim"]:
        assert len(activations.shape) == 3
        if self.seq_pos == "all":
            acts = (activations * attention_mask[:, :, None]).mean(axis=1)
        else:
            assert isinstance(self.seq_pos, int)
            assert self.seq_pos in range(activations.shape[1]), (
                f"Invalid sequence position: {self.seq_pos}"
            )
            acts = activations[:, self.seq_pos, :]

        return acts

    def _fit(
        self,
        X: Float[np.ndarray, "batch_size seq_len embed_dim"],
        y: Float[np.ndarray, " batch_size"],
        *,
        attention_mask: Float[np.ndarray, "batch_size seq_len"],
    ) -> Self:
        X = self._preprocess_activations(X, attention_mask)

        self._classifier.fit(X, y)
        return self

    def fit(self, dataset: LabelledDataset) -> Self:
        activations, attention_mask = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )

        print("Training probe...")
        self._fit(
            X=activations,
            y=dataset.labels_numpy(),
            attention_mask=attention_mask,
        )

        return self

    def predict(self, dataset: Dataset) -> list[Label]:
        activations, attention_mask = self._llm.get_activations(
            dataset.inputs, layers=[self.layer]
        )[0]
        predictions = self._predict(activations, attention_mask=attention_mask)
        return [Label.from_int(pred) for pred in predictions]

    def _predict(
        self,
        X: Float[np.ndarray, "batch_size seq_len embed_dim"],
        *,
        attention_mask: Float[np.ndarray, "batch_size seq_len"],
    ) -> Float[np.ndarray, " batch_size"]:
        X = self._preprocess_activations(X, attention_mask)
        return self._classifier.predict(X)


def compute_accuracy(
    probe: LinearProbe,
    dataset: Dataset,
    *,
    activations: Float[np.ndarray, "batch_size seq_len embed_dim"],
    attention_mask: Float[np.ndarray, "batch_size seq_len"],
) -> float:
    pred_labels = probe._predict(activations, attention_mask=attention_mask)
    return (np.array(pred_labels) == dataset.labels_numpy()).mean()
