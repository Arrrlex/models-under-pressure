import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Self, Sequence

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models_under_pressure.config import (
    GENERATED_DATASET_PATH,
    LOCAL_MODELS,
    PROBES_DIR,
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


@dataclass
class Probe(Protocol):
    _llm: LLMModel
    layer: int

    def fit(self, dataset: LabelledDataset) -> Self: ...

    def predict(self, dataset: BaseDataset) -> list[Label]: ...

    def predict_proba(
        self, dataset: BaseDataset
    ) -> Float[np.ndarray, " batch_size"]: ...

    def per_token_predictions(
        self,
        inputs: Sequence[Input],
    ) -> Float[np.ndarray, "batch_size seq_len"]: ...


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
class SklearnProbe(Probe):
    _llm: LLMModel
    layer: int

    aggregator: Aggregator

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

    def predict(self, dataset: BaseDataset) -> list[Label]:
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )
        labels = self._predict(activations_obj)
        return [Label.from_int(pred) for pred in labels]

    def predict_proba(self, dataset: BaseDataset) -> Float[np.ndarray, " batch_size"]:
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )
        return self._predict_proba(activations_obj)

    def _fit(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
    ) -> Self:
        # Preprocess the aggregations to be of the correct shape:
        X, _ = self.aggregator.preprocess(activations, y)

        self._classifier.fit(X, y)
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
        probs = self._classifier.predict_proba(X)[:, 1]
        return np.log(probs / (1 - probs))

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

        # TODO This can be done more efficiently -> so can a lot of things
        predictions = []
        for i in range(len(activations_obj.activations)):
            activations = activations_obj.activations[i]
            attention_mask = activations_obj.attention_mask[i]

            # Compute per-token predictions
            # Apply attention mask to zero out padding tokens
            X = activations * attention_mask[:, None]
            # Only keep predictions for non-zero attention mask tokens
            predicted_probs = self._classifier.predict_proba(X)[:, 1]

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
        _llm=model,
        layer=probe_info.layer,
        aggregator=probe_info.aggregator,
        _classifier=classifier,
    )


def load_or_train_probe(
    model: LLMModel,
    train_dataset: LabelledDataset,
    train_dataset_path: Path,
    layer: int,
    aggregator: Aggregator,
    seq_pos: int | str = "all",
) -> SklearnProbe:
    probe_info = ProbeInfo(
        model_name_short=model.name.split("/")[-1],
        dataset_path=train_dataset_path.stem,
        layer=layer,
        aggregator=aggregator,
        seq_pos=seq_pos,
    )
    if probe_info.path.exists():
        probe = load_probe(model, probe_info)
    else:
        probe = SklearnProbe(_llm=model, layer=layer, aggregator=aggregator).fit(
            train_dataset
        )
        save_probe(probe, probe_info)
    return probe


def compute_accuracy(
    probe: Probe,
    dataset: LabelledDataset,
    activations: Activation,
) -> float:
    pred_labels = probe.predict(dataset)
    return (np.array(pred_labels) == dataset.labels_numpy()).mean()


if __name__ == "__main__":
    model = LLMModel.load(model_name=LOCAL_MODELS["llama-1b"])

    # Train a probe
    agg = Aggregator(
        preprocessor=Preprocessors.per_token,
        postprocessor=Postprocessors.sigmoid,
    )
    train_dataset, _ = load_train_test(dataset_path=GENERATED_DATASET_PATH)
    probe = SklearnProbe(_llm=model, layer=11, aggregator=agg)
    probe.fit(train_dataset[:10])

    # Test the probe
    inputs = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    predictions = probe.per_token_predictions(inputs)
    print(predictions)
