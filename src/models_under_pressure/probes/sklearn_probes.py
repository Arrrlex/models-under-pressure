from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models_under_pressure.activation_store import ActivationStore
from models_under_pressure.config import (
    PROBES_DIR,
)
from models_under_pressure.interfaces.activations import (
    Activation,
    Aggregator,
)
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Label,
    LabelledDataset,
)
from models_under_pressure.probes.base import Classifier, Probe


@dataclass
class SklearnProbe(Probe):
    model_name: str
    layer: int

    aggregator: Aggregator

    hyper_params: dict = field(
        default_factory=lambda: {
            "C": 1e-3,
            "random_state": 42,
            "fit_intercept": False,
        }
    )
    _classifier: Classifier | None = None

    def __post_init__(self):
        if self._classifier is None:
            self._classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(**self.hyper_params),
            )  # type: ignore

    def fit(self, dataset: LabelledDataset) -> Self:
        activations_obj = ActivationStore().load(
            model_name=self.model_name,
            dataset_spec=dataset.spec,
            layer=self.layer,
        )

        print("Training probe...")
        self._fit(
            activations=activations_obj,
            y=dataset.labels_numpy(),
        )

        return self

    def predict(self, dataset: BaseDataset) -> list[Label]:
        activations = ActivationStore().load(
            model_name=self.model_name,
            dataset_spec=dataset.spec,
            layer=self.layer,
        )
        labels = self._predict(activations)
        return [Label.from_int(pred) for pred in labels]

    def predict_proba(
        self, dataset: BaseDataset
    ) -> tuple[Activation, Float[np.ndarray, " batch_size"]]:
        activations = ActivationStore().load(
            model_name=self.model_name,
            dataset_spec=dataset.spec,
            layer=self.layer,
        )
        return activations, self._predict_proba(activations)

    def predict_proba_without_activations(
        self, dataset: BaseDataset
    ) -> Float[np.ndarray, " batch_size"]:
        _, probs = self.predict_proba(dataset)
        return probs

    def _fit(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
    ) -> Self:
        # Preprocess the aggregations to be of the correct shape:
        X, _ = self.aggregator.preprocess(activations, y)

        self._classifier.fit(X, y)  # type: ignore
        return self

    def _predict(
        self,
        activations: Activation,
    ) -> Float[np.ndarray, " batch_size"]:
        return self._predict_proba(activations) > 0.5

    def _predict_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size"]:
        X, _ = self.aggregator.preprocess(activations)
        logits = self._get_logits(X)
        probs = self.aggregator.postprocess(logits)
        return probs

    def _get_logits(
        self, X: Float[np.ndarray, " batch_size ..."]
    ) -> Float[np.ndarray, " batch_size"]:
        probs = self._classifier.predict_proba(X)[:, 1]  # type: ignore
        return np.log(probs / (1 - probs))

    def per_token_predictions(
        self, dataset: BaseDataset
    ) -> Float[np.ndarray, "batch_size seq_len"]:
        activations = ActivationStore().load(
            model_name=self.model_name,
            dataset_spec=dataset.spec,
            layer=self.layer,
        )

        # TODO This can be done more efficiently -> so can a lot of things
        predictions = []
        for i in range(len(activations.get_activations(per_token=False))):
            activations = activations.get_activations(per_token=False)[i]
            attention_mask = activations.get_attention_mask(per_token=False)[i]

            # Compute per-token predictions
            # Apply attention mask to zero out padding tokens
            X = activations * attention_mask[:, None]
            # Only keep predictions for non-zero attention mask tokens
            predicted_probs = self._classifier.predict_proba(X)[:, 1]  # type: ignore

            # Set the values to -1 if they're attention masked out
            predicted_probs[attention_mask == 0] = -1

            predictions.append(predicted_probs)

        return np.array(predictions)


@dataclass
class ProbeInfo:
    model_name_short: str
    dataset_path: str
    layer: int

    aggregator: Aggregator
    seq_pos: int | str

    @property
    def name(self) -> str:
        return f"{self.model_name_short}_{self.dataset_path}_l{self.layer}_{self.aggregator.name}_{self.seq_pos}"

    @property
    def path(self) -> Path:
        return PROBES_DIR / f"{self.name}.pkl"


def compute_accuracy(
    probe: Probe,
    dataset: LabelledDataset,
) -> float:
    pred_labels = probe.predict(dataset)
    pred_labels_np = np.array([label.to_int() for label in pred_labels])
    return (pred_labels_np == dataset.labels_numpy()).mean()
