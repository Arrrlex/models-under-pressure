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
    TRAIN_TEST_SPLIT,
)
from models_under_pressure.experiments.dataset_splitting import load_train_test
from models_under_pressure.interfaces.activations import Activation, AggregationType
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Dataset,
    Input,
    Label,
    LabelledDataset,
)
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

    agg_type: AggregationType = AggregationType.MEAN
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
        activations: Activation,
    ) -> Float[np.ndarray, "batch_size embed_dim"]:
        if self.seq_pos == "all":
            if self.agg_type == AggregationType.MEAN:
                acts = activations.mean_aggregation().activations
            else:
                raise NotImplementedError(
                    f"Aggregation type {self.agg_type} not implemented"
                )
        else:
            assert isinstance(self.seq_pos, int)
            assert self.seq_pos in range(
                activations.activations.shape[1]
            ), f"Invalid sequence position: {self.seq_pos}"
            acts = activations.activations[:, self.seq_pos, :]

        return acts

    def _fit(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
    ) -> Self:
        X = self._preprocess_activations(activations)

        self._classifier.fit(X, y)
        return self

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
        predictions = self._predict(activations_obj)
        return [Label.from_int(pred) for pred in predictions]

    def predict_proba(self, dataset: BaseDataset) -> list[float]:
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )
        X = self._preprocess_activations(activations_obj)
        predictions = self._classifier.predict_proba(X)

        return predictions[:, 1].tolist()

    def _predict(
        self,
        activations: Activation,
    ) -> Float[np.ndarray, " batch_size"]:
        X = self._preprocess_activations(activations)
        return self._classifier.predict(X)

    def per_token_predictions(
        self,
        inputs: Sequence[Input],
    ) -> Float[np.ndarray, "batch_size seq_len"]:
        dataset = Dataset(
            inputs=inputs, ids=[str(i) for i in range(len(inputs))], other_fields={}
        )
        activations_obj = self._llm.get_batched_activations(
            dataset=dataset,
            layer=self.layer,
        )

        # TODO This can be done more efficiently
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

    agg_type: AggregationType
    seq_pos: int | str

    @property
    def name(self) -> str:
        return f"{self.model_name_short}_{self.dataset_path}_l{self.layer}_{self.agg_type}_{self.seq_pos}"

    @property
    def path(self) -> Path:
        return PROBES_DIR / f"{self.name}.pkl"


def save_probe(probe: LinearProbe, probe_info: ProbeInfo):
    output_path = probe_info.path

    print(f"Saving probe to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(probe._classifier, f)


def load_probe(model: LLMModel, probe_info: ProbeInfo) -> LinearProbe:
    probe_path = probe_info.path
    print(f"Loading probe from {probe_path}")
    with open(probe_path, "rb") as f:
        classifier = pickle.load(f)
    return LinearProbe(
        _llm=model,
        layer=probe_info.layer,
        agg_type=probe_info.agg_type,
        seq_pos=probe_info.seq_pos,
        _classifier=classifier,
    )


def load_or_train_probe(
    model: LLMModel,
    train_dataset: LabelledDataset,
    train_dataset_path: Path,
    layer: int,
    agg_type: AggregationType = AggregationType.MEAN,
    seq_pos: int | str = "all",
) -> LinearProbe:
    probe_info = ProbeInfo(
        model_name_short=model.name.split("/")[-1],
        dataset_path=train_dataset_path.stem,
        layer=layer,
        agg_type=agg_type,
        seq_pos=seq_pos,
    )
    if probe_info.path.exists():
        probe = load_probe(model, probe_info)
    else:
        probe = LinearProbe(_llm=model, layer=layer).fit(train_dataset)
        save_probe(probe, probe_info)
    return probe


def compute_accuracy(
    probe: LinearProbe,
    dataset: LabelledDataset,
    activations: Activation,
) -> float:
    pred_labels = probe._predict(activations)
    return (np.array(pred_labels) == dataset.labels_numpy()).mean()


if __name__ == "__main__":
    model = LLMModel.load(model_name=LOCAL_MODELS["llama-8b"])

    # Train a probe
    train_dataset, _ = load_train_test(
        dataset_path=GENERATED_DATASET_PATH,
        split_path=TRAIN_TEST_SPLIT,
    )
    probe = LinearProbe(_llm=model, layer=11)
    probe.fit(train_dataset[:10])

    # Test the probe
    inputs = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    predictions = probe.per_token_predictions(inputs)
    print(predictions)
