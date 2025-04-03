from dataclasses import dataclass
from typing import Self

import numpy as np
from jaxtyping import Float

from models_under_pressure.activation_store import ActivationStore
from models_under_pressure.interfaces.activations import (
    Activation,
)
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Label,
    LabelledDataset,
)
from models_under_pressure.probes.pytorch_classifiers import PytorchLinearClassifier
from models_under_pressure.probes.sklearn_probes import Probe


@dataclass
class PytorchProbe(Probe):
    model_name: str
    layer: int

    hyper_params: dict
    _classifier: PytorchLinearClassifier | None = None

    def __post_init__(self):
        if self._classifier is None:
            self._classifier = PytorchLinearClassifier(training_args=self.hyper_params)

    def fit(self, dataset: LabelledDataset) -> Self:
        """
        Fit the probe to the dataset, return a self object with a trained classifier.
        """
        activations = ActivationStore().load(
            model_name=self.model_name,
            dataset_spec=dataset.spec,
            layer=self.layer,
        )

        print("Training probe...")
        self._classifier = self._fit(
            activations=activations,
            y=dataset.labels_numpy(),
        )

        return self

    def _fit(self, activations: Activation, y: Float[np.ndarray, " batch_size"]):
        # Pass the activations and labels to the pytorch classifier class:
        return self._classifier.train(activations, y)  # type: ignore

    def predict(self, dataset: BaseDataset) -> list:
        """
        Predict and return the labels of the dataset.
        """
        activations = ActivationStore().load(
            model_name=self.model_name,
            dataset_spec=dataset.spec,
            layer=self.layer,
        )

        labels = self._classifier.predict(activations)  # type: ignore
        return [Label.from_int(label) for label in labels]

    def predict_proba(
        self, dataset: BaseDataset
    ) -> tuple[Activation, Float[np.ndarray, " batch_size"]]:
        """
        Predict and return the probabilities of the dataset.

        Probabilities are expected from the classifier in the shape (batch_size,)
        """
        activations = ActivationStore().load(
            model_name=self.model_name,
            dataset_spec=dataset.spec,
            layer=self.layer,
        )

        # Get the batch_size, seq_len probabilities:
        probs = self._classifier.predict_proba(activations)  # type: ignore

        # Take the mean over the sequence length:
        return activations, probs

    def per_token_predictions(
        self,
        dataset: LabelledDataset,
    ) -> Float[np.ndarray, "batch_size seq_len"]:
        """
        Probabilities are expected in the shape (batch_size, seq_len) by the classifier.
        """

        # TODO: Change such that it uses the aggregation framework
        activations = ActivationStore().load(
            model_name=self.model_name,
            dataset_spec=dataset.spec,
            layer=self.layer,
        )

        return self._classifier.predict_token_proba(activations)  # type: ignore
