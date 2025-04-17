from dataclasses import dataclass, field
from typing import Self, Sequence

import numpy as np
from jaxtyping import Float

from models_under_pressure.config import (
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
)
from models_under_pressure.dataset_utils import load_train_test
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
    _classifier: PytorchLinearClassifier = field(
        default_factory=lambda: PytorchLinearClassifier(training_args={})
    )

    def __post_init__(self):
        self._classifier.training_args = self.hyper_params

    def fit(self, dataset: LabelledDataset) -> Self:
        """
        Fit the probe to the dataset, return a self object with a trained classifier.
        """
        activations_obj = Activation.from_dataset(dataset)

        print("Training probe...")
        self._classifier.train(activations_obj, dataset.labels_numpy())
        return self

    def predict(self, dataset: BaseDataset) -> list:
        """
        Predict and return the labels of the dataset.
        """

        activations_obj = Activation.from_dataset(dataset)
        labels = self._classifier.predict(activations_obj)
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
        probs = self._classifier.predict_proba(activations_obj)

        # Take the mean over the sequence length:
        return activations_obj, probs

    def predict_proba_without_activations(
        self, dataset: BaseDataset
    ) -> Float[np.ndarray, " batch_size"]:
        """
        Predict and return the probabilities of the dataset.

        Probabilities are expected from the classifier in the shape (batch_size,)
        """
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )

        # Get the batch_size, seq_len probabilities:
        return self._classifier.predict_proba(activations_obj)  # type: ignore

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

        probs = self._classifier.predict_token_proba(activations_obj)

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
