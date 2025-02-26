# Imports
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from models_under_pressure.interfaces.dataset import Dataset, Label
from models_under_pressure.probes.model import LLMModel


class HighStakesClassifier(Protocol):
    def predict(self, dataset: Dataset) -> list[Label]: ...


@dataclass
class LinearProbeClassifier:
    _llm: LLMModel
    _classifier: Any = field(init=False)
    seq_pos: int | str = "all"
    _classifier_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"C": 1e-3, "random_state": 42, "fit_intercept": False}
    )

    def __post_init__(self):
        self._classifier = make_pipeline(
            StandardScaler(), LogisticRegression(**self._classifier_kwargs)
        )

    def _preprocess_activations(
        self,
        activations: Float[np.ndarray, "batch_size seq_len embed_dim"],
    ) -> Float[np.ndarray, "batch_size embed_dim"]:
        if self.seq_pos == "all":
            acts = activations.mean(axis=1)
        else:
            assert isinstance(self.seq_pos, int)
            assert self.seq_pos in range(
                activations.shape[1]
            ), f"Invalid sequence position: {self.seq_pos}"
            acts = activations[:, self.seq_pos, :]

        return acts

    def fit(
        self,
        X: Float[np.ndarray, "batch_size seq_len embed_dim"],
        y: Float[np.ndarray, " batch_size"],
    ) -> "LinearProbe":
        X = self._preprocess_activations(X)

        self._classifier.fit(X, y)
        return self

    def predict(
        self, X: Float[np.ndarray, "batch_size seq_len embed_dim"]
    ) -> Float[np.ndarray, " batch_size"]:
        X = self._preprocess_activations(X)
        return self._model.predict(X)


@torch.no_grad()
def create_activations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    text: list[str],
    device: str | torch.device,
) -> Float[np.ndarray, "layers batch_size seq_len embed_dim"]:
    # Tokenize input text
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=1028
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Dictionary to store residual activations
    activations = []

    # Hook function to capture residual activations before layernorm
    def hook_fn(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        activations.append(input[0].detach().cpu())  # Store the residual connection

    # Register hooks on each transformer block (LLaMA layers)
    hooks: list[torch.nn.Module] = [
        # Pre-attention residual
        layer.input_layernorm.register_forward_hook(hook_fn)
        for layer in model.model.layers
    ]

    # Forward pass
    _ = model(**inputs)

    # Remove hooks after capturing activations
    for hook in hooks:
        hook.remove()

    # Print stored activations
    for i, act in enumerate(activations):
        print(f"Layer: {i}, Activation Shape: {act.shape}")

    all_acts = torch.stack(activations)
    print("All activations shape:", all_acts.shape)

    return all_acts.cpu().detach().numpy()


@dataclass
class LinearProbe:
    seq_pos: int | str = "all"
    model_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"C": 1e-3, "random_state": 42, "fit_intercept": False}
    )
    _model: Any = field(init=False)

    def __post_init__(self):
        self._model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**self.model_kwargs)),
            ]
        )

    def _preprocess_activations(
        self,
        activations: Float[np.ndarray, "batch_size seq_len embed_dim"],
    ) -> Float[np.ndarray, "batch_size embed_dim"]:
        if self.seq_pos == "all":
            acts = activations.mean(axis=1)
        else:
            assert isinstance(self.seq_pos, int)
            assert self.seq_pos in range(
                activations.shape[1]
            ), f"Invalid sequence position: {self.seq_pos}"
            acts = activations[:, self.seq_pos, :]

        return acts

    def fit(
        self,
        X: Float[np.ndarray, "batch_size seq_len embed_dim"],
        y: Float[np.ndarray, " batch_size"],
    ) -> "LinearProbe":
        X = self._preprocess_activations(X)

        self._model.fit(X, y)
        return self

    def predict(
        self, X: Float[np.ndarray, "batch_size seq_len embed_dim"]
    ) -> Float[np.ndarray, " batch_size"]:
        X = self._preprocess_activations(X)
        return self._model.predict(X)


def train_single_layer(
    acts: Float[np.ndarray, "batch_size seq_len embed_dim"],
    labels: Float[np.ndarray, " batch_size"],
    model_params: dict[str, Any] | None = None,
) -> LinearProbe:
    if model_params is None:
        probe = LinearProbe()
    else:
        probe = LinearProbe(model_kwargs=model_params)

    return probe.fit(acts, labels)


def compute_accuracy(
    probe: LinearProbe,
    activations: Float[np.ndarray, "batch_size seq_len embed_dim"],
    labels: Float[np.ndarray, " batch_size"] | list[int],
):
    pred_labels = probe.predict(activations)
    return (pred_labels == labels).mean()
