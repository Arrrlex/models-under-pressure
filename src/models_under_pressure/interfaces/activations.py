from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import Dataset as TorchDataset

from models_under_pressure.config import global_settings
from models_under_pressure.interfaces.dataset import BaseDataset
from models_under_pressure.utils import as_numpy


class ActivationDataset(TorchDataset):
    """
    A pytorch Dataset class that contains the activations structured as a batch-wise dataset.

    This dataset can be either
    - per-token (activations shape: (b, s, e), attention_mask shape: (b, s))
    - or per-entry (activations shape: (b * s, e), attention_mask shape: (b * s))

    where b is the batch size, s is the sequence length, and e is the embedding dimension.
    """

    def __init__(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        y: torch.Tensor,
        device: torch.device | str = global_settings.DEVICE,
        dtype: torch.dtype = global_settings.DTYPE,
    ):
        self.activations = activations
        self.attention_mask = attention_mask
        self.input_ids = input_ids
        self.y = y
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the masked activations, attention mask, input ids and label.
        """
        return self.__getitems__([index])[0]

    def __getitems__(
        self, indices: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Get the tensors for the batch indices
        batch_acts = self.activations[indices].to(self.device).to(self.dtype)
        batch_mask = self.attention_mask[indices].to(self.device).to(self.dtype)
        batch_input_ids = self.input_ids[indices].to(self.device).to(self.dtype)
        batch_y = self.y[indices].to(self.device).to(self.dtype)

        # Return as a list of tuples
        return [
            (batch_acts[i], batch_mask[i], batch_input_ids[i], batch_y[i])
            for i in range(len(indices))
        ]


@dataclass
class Activation:
    activations: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    attention_mask: Float[torch.Tensor, "batch_size seq_len"]
    input_ids: Float[torch.Tensor, "batch_size seq_len"]

    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> "Activation":
        return cls(
            activations=dataset.other_fields["activations"],  # type: ignore
            attention_mask=dataset.other_fields["attention_mask"],  # type: ignore
            input_ids=dataset.other_fields["input_ids"],  # type: ignore
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.activations.shape

    @property
    def batch_size(self) -> int:
        return self.activations.shape[0]

    @property
    def seq_len(self) -> int:
        return self.activations.shape[1]

    @property
    def embed_dim(self) -> int:
        return self.activations.shape[2]

    def __post_init__(self):
        """Validate shapes after initialization, applies attention mask to activations."""
        shape = (self.batch_size, self.seq_len)
        assert (
            self.attention_mask.shape == shape
        ), f"Attention mask shape {self.attention_mask.shape} doesn't agree with {shape}"
        assert (
            self.input_ids.shape == shape
        ), f"Input ids shape {self.input_ids.shape} doesn't agree with {shape}"

        self.activations *= self.attention_mask[:, :, None]

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "Activation":
        return Activation(
            activations=self.activations.to(device).to(dtype),
            attention_mask=self.attention_mask.to(device).to(dtype),
            input_ids=self.input_ids.to(device).to(dtype),
        )

    def per_token(self) -> "PerTokenActivation":
        activations = einops.rearrange(self.activations, "b s e -> (b s) e")
        attention_mask = einops.rearrange(self.attention_mask, "b s -> (b s)")
        input_ids = einops.rearrange(self.input_ids, "b s -> (b s)")
        return PerTokenActivation(
            activations=activations, attention_mask=attention_mask, input_ids=input_ids
        )

    def to_dataset(self, y: Float[torch.Tensor, " batch_size"]) -> ActivationDataset:
        return ActivationDataset(
            activations=self.activations,
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
            y=y,
        )


@dataclass
class PerTokenActivation:
    activations: Float[torch.Tensor, "tokens embed_dim"]
    attention_mask: Float[torch.Tensor, " tokens"]
    input_ids: Float[torch.Tensor, " tokens"]

    def to_dataset(self, y: Float[torch.Tensor, " batch_size"]) -> ActivationDataset:
        tokens = self.activations.shape[0]
        seq_len, rem = divmod(tokens, y.shape[0])
        assert (
            rem == 0
        ), f"Batch size {y.shape[0]} does not divide the number of tokens {tokens}"
        y = einops.repeat(y, "b -> (b s)", s=seq_len)
        return ActivationDataset(
            activations=self.activations,
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
            y=y,
        )


class Preprocessor(Protocol):
    def __call__(
        self, X: Activation, y: Optional[Float[np.ndarray, " batch_size"]] = None
    ) -> tuple[
        Float[np.ndarray, "flattened_batch_size embed_dim"],
        Optional[Float[np.ndarray, " flattened_batch_size"]],
    ]: ...


class Postprocessor(Protocol):
    def __call__(
        self,
        logits: Float[np.ndarray, "flattened_batch_size ..."],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " new_batch_size"]: ...


class Preprocessors:
    @staticmethod
    def mean(
        X: Activation,
        y: Optional[Float[np.ndarray, " batch_size"]] = None,
        batch_size: int = 200,
    ) -> Tuple[
        Float[np.ndarray, " batch_size embed_dim"],
        Optional[Float[np.ndarray, " batch_size"]],
    ]:
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
        return (sum_acts / token_counts), y


class Postprocessors:
    @staticmethod
    def sigmoid(
        logits: Float[np.ndarray, " batch_size"],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " batch_size"]:
        return 1 / (1 + np.exp(-logits))


@dataclass
class Aggregator:
    preprocessor: Preprocessor
    postprocessor: Postprocessor
    original_shape: tuple[int, int, int] | None = None

    def preprocess(
        self, X: Activation, y: Optional[Float[np.ndarray, " batch_size"]] = None
    ) -> tuple[
        Float[np.ndarray, " batch_size ..."], Optional[Float[np.ndarray, " batch_size"]]
    ]:
        self.original_shape = X.shape  # type: ignore
        return self.preprocessor(X, y)

    def postprocess(
        self, logits: Float[np.ndarray, " flattened_batch_size ..."]
    ) -> Float[np.ndarray, " batch_size"]:
        if self.original_shape is None:
            raise ValueError("Original shape not set")
        return self.postprocessor(logits, self.original_shape)

    @property
    def name(self) -> str:
        if hasattr(self.preprocessor, "id"):
            preprocessor_id = self.preprocessor.id  # type: ignore
        else:
            preprocessor_id = self.preprocessor.__class__.__name__
        if hasattr(self.postprocessor, "id"):
            postprocessor_id = self.postprocessor.id  # type: ignore
        else:
            postprocessor_id = self.postprocessor.__class__.__name__
        return f"{preprocessor_id}_{postprocessor_id}"
