from dataclasses import dataclass
from typing import Any, Self, Sequence

import numpy as np
from jaxtyping import Float

from models_under_pressure.interfaces.activations import (
    Activation,
)
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Dataset,
    Input,
    Label,
    LabelledDataset,
)
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchClassifier,
)
from models_under_pressure.probes.sklearn_probes import Probe
from models_under_pressure.utils import as_numpy


@dataclass
class PytorchProbe(Probe):
    hyper_params: dict
    _classifier: PytorchClassifier

    def __post_init__(self):
        self._classifier.training_args = self.hyper_params

    def fit(
        self,
        dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None = None,
        **train_args: Any,
    ) -> Self:
        """
        Fit the probe to the dataset, return a self object with a trained classifier.
        """
        activations_obj = Activation.from_dataset(dataset)

        print("Training probe...")
        if validation_dataset is not None:
            self._classifier.train(
                activations_obj,
                dataset.labels_torch(),
                validation_activations=Activation.from_dataset(validation_dataset),
                validation_y=validation_dataset.labels_torch(),
                **train_args,
            )
        else:
            self._classifier.train(
                activations_obj, dataset.labels_torch(), **train_args
            )
        return self

    def predict(self, dataset: BaseDataset) -> list[Label]:
        """
        Predict and return the labels of the dataset.
        """

        labels = self.predict_proba(dataset) > 0.5
        return [Label.from_int(label) for label in labels]

    def predict_proba(self, dataset: BaseDataset) -> Float[np.ndarray, " batch_size"]:
        """
        Predict and return the probabilities of the dataset.

        Probabilities are expected from the classifier in the shape (batch_size,)
        """
        activations_obj = Activation.from_dataset(dataset)
        return as_numpy(self._classifier.probs(activations_obj))

    def per_token_predictions(
        self,
        inputs: Sequence[Input],
    ) -> Float[np.ndarray, "batch_size seq_len"]:
        dataset = Dataset(
            inputs=inputs, ids=[str(i) for i in range(len(inputs))], other_fields={}
        )
        """
        Probabilities are expected in the shape (batch_size, seq_len) by the classifier.
        """

        # TODO: Change such that it uses the aggregation framework
        activations_obj = Activation.from_dataset(dataset)

        probs = self._classifier.probs(activations_obj, per_token=True)

        return as_numpy(probs)
