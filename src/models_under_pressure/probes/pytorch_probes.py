from dataclasses import dataclass
from typing import Self, Sequence

import numpy as np
from jaxtyping import Float

from models_under_pressure.config import (
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
)
from models_under_pressure.experiments.dataset_splitting import load_train_test
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
from models_under_pressure.probes.pytorch_classifiers import PytorchLinearClassifier
from models_under_pressure.probes.sklearn_probes import Probe


@dataclass
class PytorchProbe(Probe):
    hyper_params: dict
    _classifier: PytorchLinearClassifier | None = None

    def __post_init__(self):
        if self._classifier is None:
            self._classifier = PytorchLinearClassifier(training_args=self.hyper_params)

    def fit(self, dataset: LabelledDataset) -> Self:
        """
        Fit the probe to the dataset, return a self object with a trained classifier.
        """
        activations_obj = Activation.from_dataset(dataset)

        print("Training probe...")
        self._classifier = self._fit(
            activations=activations_obj,
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
        activations_obj = Activation.from_dataset(dataset)
        labels = self._classifier.predict(activations_obj)  # type: ignore
        return [Label.from_int(label) for label in labels]

    def predict_proba(
        self, dataset: BaseDataset
    ) -> tuple[Activation, Float[np.ndarray, " batch_size"]]:
        """
        Predict and return the probabilities of the dataset.

        Probabilities are expected from the classifier in the shape (batch_size,)
        """
        activations_obj = Activation.from_dataset(dataset)

        # Get the batch_size, seq_len probabilities:
        probs = self._classifier.predict_proba(activations_obj)  # type: ignore

        # Take the mean over the sequence length:
        return activations_obj, probs

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

        probs = self._classifier.predict_token_proba(activations_obj)  # type: ignore

        return probs


if __name__ == "__main__":
    # Train a probe
    train_dataset, _ = load_train_test(
        dataset_path=SYNTHETIC_DATASET_PATH,
        model_name=LOCAL_MODELS["llama-1b"],
        layer=11,
    )
    hyper_params = {
        "batch_size": 16,
        "epochs": 3,
        "device": "cpu",
    }
    probe = PytorchProbe(hyper_params=hyper_params)
    probe.fit(train_dataset[:10])

    # Test the probe
    inputs = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    predictions = probe.per_token_predictions(inputs)
    print(predictions)
