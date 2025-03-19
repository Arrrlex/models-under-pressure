from dataclasses import dataclass, field
from typing import Protocol, Self, Sequence

import numpy as np
from jaxtyping import Float

from models_under_pressure.config import (
    GENERATED_DATASET_PATH,
    LOCAL_MODELS,
)
from models_under_pressure.experiments.dataset_splitting import load_train_test
from models_under_pressure.interfaces.activations import (
    Activation,
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Dataset,
    Input,
    Label,
    LabelledDataset,
)
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.pytorch_classifiers import PytorchLinearClassifier


class HighStakesClassifier(Protocol):
    def predict(self, dataset: Dataset) -> list[Label]: ...


@dataclass
class PytorchProbe(HighStakesClassifier):
    _llm: LLMModel
    layer: int
    aggregator: Aggregator | None = None
    _classifier: PytorchLinearClassifier = field(
        default_factory=PytorchLinearClassifier
    )

    def fit(self, dataset: LabelledDataset) -> Self:
        """
        Fit the probe to the dataset, return a self object with a trained classifier.
        """
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )

        print("Training probe...")
        self._classifier = self._fit(
            activations=activations_obj,
            y=dataset.labels_numpy(),
        )

        return self

    def _fit(self, activations: Activation, y: Float[np.ndarray, " batch_size"]):
        # Pass the activations and labels to the pytorch classifier class:
        return self._classifier.train(activations, y)

    def predict(self, dataset: BaseDataset) -> list:
        """
        Predict and return the labels of the dataset.
        """
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )
        labels = self._classifier.predict(activations_obj)
        return [Label.from_int(label) for label in labels]

    def predict_proba(self, dataset: BaseDataset) -> Float[np.ndarray, " batch_size"]:
        """
        Predict and return the probabilities of the dataset.
        """
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )
        return self._classifier.predict_proba(activations_obj)

    def per_token_predictions(
        self,
        inputs: Sequence[Input],
    ) -> Float[np.ndarray, "batch_size seq_len"]:
        dataset = Dataset(
            inputs=inputs, ids=[str(i) for i in range(len(inputs))], other_fields={}
        )

        # TODO: Change such that it uses the aggregation framework

        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )

        probs = self._classifier.predict_proba(activations_obj)

        return probs


if __name__ == "__main__":
    model = LLMModel.load(model_name=LOCAL_MODELS["llama-1b"])

    # Train a probe
    agg = Aggregator(
        preprocessor=Preprocessors.per_token,
        postprocessor=Postprocessors.sigmoid,
    )
    train_dataset, _ = load_train_test(dataset_path=GENERATED_DATASET_PATH)
    probe = PytorchProbe(_llm=model, layer=11)
    probe.fit(train_dataset[:10])

    # Test the probe
    inputs = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    predictions = probe.per_token_predictions(inputs)
    print(predictions)
