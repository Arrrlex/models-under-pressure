import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Sequence

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models_under_pressure.config import (
    LOCAL_MODELS,
    PROBES_DIR,
    SYNTHETIC_DATASET_PATH,
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
from models_under_pressure.model import LLMModel
from models_under_pressure.probes.base import Classifier, Probe


@dataclass
class SklearnProbe(Probe):
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

    def predict_proba(
        self, dataset: BaseDataset
    ) -> tuple[Activation, Float[np.ndarray, " batch_size"]]:
        activations_obj = Activation.from_dataset(dataset)
        return activations_obj, self._predict_proba(activations_obj)

    def predict_proba_without_activations(
        self, dataset: BaseDataset
    ) -> Float[np.ndarray, " batch_size"]:
        activations_obj = Activation.from_dataset(dataset)
        # print(
        #     f"DEBUGGING: Obtained {len(activations_obj.get_activations(per_token=False))} activations"
        # )
        return self._predict_proba(activations_obj)

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
        self,
        inputs: Sequence[Input],
    ) -> Float[np.ndarray, "batch_size seq_len"]:
        dataset = Dataset(
            inputs=inputs, ids=[str(i) for i in range(len(inputs))], other_fields={}
        )

        # TODO: Change such that it uses the aggregation framework

        activations_obj = Activation.from_dataset(dataset)

        # TODO This can be done more efficiently -> so can a lot of things
        predictions = []
        for i in range(len(activations_obj.get_activations(per_token=False))):
            activations = activations_obj.get_activations(per_token=False)[i]
            attention_mask = activations_obj.get_attention_mask(per_token=False)[i]

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


def save_probe(probe: SklearnProbe, probe_info: ProbeInfo):
    output_path = probe_info.path

    print(f"Saving probe to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(probe._classifier, f)


def load_probe(model: LLMModel, probe_info: ProbeInfo) -> SklearnProbe:
    probe_path = probe_info.path
    print(f"Loading probe from {probe_path}")
    with open(probe_path, "rb") as f:
        classifier = pickle.load(f)
    return SklearnProbe(
        aggregator=probe_info.aggregator,
        _classifier=classifier,
    )


def compute_accuracy(
    probe: Probe,
    dataset: LabelledDataset,
) -> float:
    pred_labels = probe.predict(dataset)
    pred_labels_np = np.array([label.to_int() for label in pred_labels])
    return (pred_labels_np == dataset.labels_numpy()).mean()


if __name__ == "__main__":
    model = LLMModel.load(LOCAL_MODELS["llama-1b"])

    # Train a probe
    agg = Aggregator(
        preprocessor=Preprocessors.per_token,
        postprocessor=Postprocessors.sigmoid,
    )
    train_dataset, _ = load_train_test(dataset_path=SYNTHETIC_DATASET_PATH)
    probe = SklearnProbe(layer=11, aggregator=agg)
    probe.fit(train_dataset[:10])

    # Test the probe
    inputs = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    predictions = probe.per_token_predictions(inputs)
    print(predictions)
