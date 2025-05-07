from dataclasses import dataclass, field
from typing import Literal, Self, Sequence

import numpy as np
from jaxtyping import Float
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Dataset,
    Input,
    Label,
    LabelledDataset,
)
from models_under_pressure.probes.base import Classifier, Probe
from models_under_pressure.utils import as_numpy


@dataclass
class SklearnProbe(Probe):
    probe_type: Literal["logistic_regression", "GaussianProcessClassifier"] = (
        "logistic_regression"
    )
    hyper_params: dict = field(
        default_factory=lambda: {
            "C": 1e-3,
            "random_state": 42,
            "fit_intercept": False,
        }
    )
    _classifier: Classifier | None = None

    def __post_init__(self):
        """
        Create the sklearn classifier model to be optimized.
        """
        if self._classifier is None:
            if self.probe_type == "logistic_regression":
                self._classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(**self.hyper_params),
                )  # type: ignore
            elif self.probe_type == "GaussianProcessClassifier":
                self._classifier = make_pipeline(
                    StandardScaler(),
                    GaussianProcessClassifier(**self.hyper_params),
                )  # type: ignore
            else:
                raise ValueError(f"Invalid probe type: {self.probe_type}")

    def fit(
        self,
        dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None = None,
    ) -> Self:
        if validation_dataset is not None:
            print("Warning: SklearnProbe does not use a validation dataset")
        activations_obj = Activation.from_dataset(dataset)

        print("Training probe...")
        self._fit(
            activations=activations_obj,
            y=dataset.labels_numpy(),
        )

        return self

    def predict(self, dataset: BaseDataset) -> list[Label]:
        activations_obj = Activation.from_dataset(dataset)
        labels = self._predict(activations_obj)
        return [Label.from_int(pred) for pred in labels]

    def predict_proba(self, dataset: BaseDataset) -> Float[np.ndarray, " batch_size"]:
        activations_obj = Activation.from_dataset(dataset)
        return self._predict_proba(activations_obj)

    def _fit(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
    ) -> Self:
        X = mean_acts(activations)
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
        X = mean_acts(activations)
        logits = self._get_logits(X)
        probs = sigmoid(logits)
        return probs

    def _get_logits(
        self, X: Float[np.ndarray, " batch_size ..."]
    ) -> Float[np.ndarray, " batch_size"]:
        probs = self._classifier.predict_proba(X)[:, 1]  # type: ignore
        return np.log(probs / (1 - probs))

    def per_token_predictions(
        self,
        inputs: Sequence[Input],
    ) -> Float[np.ndarray, "batch_size seq_len"]:
        dataset = Dataset(
            inputs=inputs, ids=[str(i) for i in range(len(inputs))], other_fields={}
        )

        # TODO: Change such that it uses the aggregation framework

        activations_obj = Activation.from_dataset(dataset)
        acts = as_numpy(activations_obj.activations)
        attn_mask = as_numpy(activations_obj.attention_mask)

        # TODO This can be done more efficiently -> so can a lot of things
        predictions = []
        for i in range(len(acts)):
            activations = acts[i]
            attention_mask = attn_mask[i]

            # Compute per-token predictions
            # Apply attention mask to zero out padding tokens
            X = activations * attention_mask[:, None]
            # Only keep predictions for non-zero attention mask tokens
            predicted_probs = self._classifier.predict_proba(X)[:, 1]  # type: ignore

            # Set the values to -1 if they're attention masked out
            predicted_probs[attention_mask == 0] = -1

            predictions.append(predicted_probs)

        return np.array(predictions)


def compute_accuracy(
    probe: Probe,
    dataset: LabelledDataset,
) -> float:
    pred_labels = probe.predict(dataset)
    pred_labels_np = np.array([label.to_int() for label in pred_labels])
    return (pred_labels_np == dataset.labels_numpy()).mean()


def mean_acts(
    X: Activation,
    batch_size: int = 200,
) -> Float[np.ndarray, " batch_size embed_dim"]:
    # Initialize accumulators for sum and token counts
    sum_acts = np.zeros((X.batch_size, X.embed_dim))
    token_counts = np.zeros((X.batch_size, 1))

    # Process in batches
    for i in range(0, X.batch_size, batch_size):
        end_idx = min(i + batch_size, X.batch_size)

        # Get current batch
        batch_acts = as_numpy(X.activations[i:end_idx])
        batch_mask = as_numpy(X.attention_mask[i:end_idx])

        # Process current batch in-place
        sum_acts[i:end_idx] = batch_acts.sum(axis=1)
        token_counts[i:end_idx] = batch_mask.sum(axis=1, keepdims=True)

    # Add small epsilon to avoid division by zero
    token_counts = token_counts + 1e-10

    # Calculate final mean
    return sum_acts / token_counts


def sigmoid(
    logits: Float[np.ndarray, " batch_size"],
) -> Float[np.ndarray, " batch_size"]:
    return 1 / (1 + np.exp(-logits))
