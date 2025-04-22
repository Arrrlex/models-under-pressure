from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import Dataset as TorchDataset

from models_under_pressure.config import global_settings
from models_under_pressure.interfaces.dataset import BaseDataset


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
    ):
        self.activations = activations
        self.attention_mask = attention_mask
        self.input_ids = input_ids
        self.y = y

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the masked activations, attention mask, input ids and label.
        """

        return tuple(
            item.to(global_settings.DEVICE).to(global_settings.DTYPE)
            for item in [
                self.activations[index],
                self.attention_mask[index],
                self.input_ids[index],
                self.y[index],
            ]
        )  # type: ignore


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
        self.shape = self.activations.shape
        self.batch_size, self.seq_len, self.embed_dim = self.shape

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


class RollingMean(Preprocessor):
    """
    Preprocessor that computes the rolling mean of the activations over a window of the specified size.

    Args:
        window_size: The size of the window to compute the rolling mean over.

    Returns:
        A tuple containing the rolling mean of the activations and the original labels,
        the shape of the rolling mean is (batch_size, shortened_seq_len, embed_dim) where
        shortened_seq_len = seq_len - window_size + 1 i.e. the number of windows that fit in the sequence.
    """

    window_size: int

    def __call__(
        self,
        X: Activation,
        y: Optional[Float[np.ndarray, " batch_size"]] = None,
    ) -> Tuple[
        Float[np.ndarray, "batch_size shortened_seq_len embed_dim"],
        Optional[Float[np.ndarray, " batch_size"]],
    ]:
        batch_size, seq_len, embed_dim = X._activations.shape
        shortened_seq_len = seq_len - self.window_size + 1

        # If sequence is shorter than window_size, return empty array
        if shortened_seq_len <= 0:
            # TODO: Return the mean and a warning?
            return np.zeros((batch_size, 0, embed_dim)), y

        result = np.zeros((batch_size, shortened_seq_len, embed_dim))

        for i in range(shortened_seq_len):
            # Window now always has full size
            start_idx = i
            end_idx = i + self.window_size

            # Get the window of activations
            window_acts = X._activations[:, start_idx:end_idx, :]
            # Get corresponding attention mask values
            window_mask = X._attention_mask[:, start_idx:end_idx, None]

            # Apply mask to zero out padding tokens
            masked_window = window_acts * window_mask

            # Count valid (non-padding) tokens in window
            valid_tokens = window_mask.sum(axis=1) + 1e-10  # epsilon to avoid div by 0

            # Sum masked values and divide by number of valid tokens
            result[:, i, :] = masked_window.sum(axis=1) / valid_tokens

        raise NotImplementedError("y reshaping not implemented")
        return result.reshape(batch_size * shortened_seq_len, embed_dim), y

    @property
    def id(self) -> str:
        return f"rolling_mean_{self.window_size}"


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
        batch_size_total, seq_len, embed_dim = X._activations.shape
        sum_acts = torch.zeros((batch_size_total, embed_dim))
        token_counts = torch.zeros((batch_size_total, 1))

        # Process in batches
        for i in range(0, batch_size_total, batch_size):
            end_idx = min(i + batch_size, batch_size_total)

            # Get current batch
            batch_acts = X._activations[i:end_idx].float()
            batch_mask = X._attention_mask[i:end_idx]

            # Process current batch in-place
            batch_acts *= batch_mask[:, :, None]  # In-place multiplication
            sum_acts[i:end_idx] = batch_acts.sum(dim=1)
            token_counts[i:end_idx] = batch_mask.sum(dim=1, keepdim=True)

        # Add small epsilon to avoid division by zero
        token_counts = token_counts + 1e-10

        # Calculate final mean
        return (sum_acts / token_counts).numpy(), y

    @staticmethod
    def per_token(
        X: Activation, y: Optional[Float[np.ndarray, " batch_size"]] = None
    ) -> Tuple[
        Float[np.ndarray, "batch_size * seq_len embed_dim"],
        Optional[Float[np.ndarray, " batch_size * seq_len"]],
    ]:
        _, seq_len, _ = X._activations.shape

        # Shape: (batch_size, seq_len, embed_dim)
        masked_acts = X._activations * X._attention_mask[:, :, None]

        # Repeat y to match flattened sequence length
        if y is not None:
            y = einops.repeat(y, "b -> (b s)", s=seq_len)

        # Shape: (batch_size * seq_len, embed_dim)
        return einops.rearrange(masked_acts, "b s e -> (b s) e"), y


class Postprocessors:
    @staticmethod
    def _reshape(
        logits: Float[np.ndarray, "flattened_batch_size embed_dim"],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " batch_size embed_dim"]:
        batch_size, _, embed_dim = original_shape
        return logits.reshape(batch_size, -1, embed_dim)

    @staticmethod
    def sigmoid(
        logits: Float[np.ndarray, " batch_size"],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " batch_size"]:
        return 1 / (1 + np.exp(-logits))

    @staticmethod
    def identity(
        logits: Float[np.ndarray, "flattened_batch_size embed_dim"],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " batch_size embed_dim"]:
        return logits.reshape(original_shape)

    @staticmethod
    def mean(
        logits: Float[np.ndarray, "flattened_batch_size embed_dim"],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " batch_size"]:
        reshaped_logits = Postprocessors._reshape(logits, original_shape)
        return Postprocessors.sigmoid(reshaped_logits.mean(axis=1), original_shape)

    @staticmethod
    def max(
        logits: Float[np.ndarray, "flattened_batch_size seq_len ..."],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " batch_size"]:
        reshaped_logits = Postprocessors._reshape(logits, original_shape)
        return Postprocessors.sigmoid(reshaped_logits.max(axis=1), original_shape)

    @staticmethod
    def max_of_rolling_mean(window_size: int) -> Postprocessor:
        return MaxOfRollingMean(window_size=window_size)


@dataclass
class MaxOfRollingMean:
    window_size: int

    def __call__(
        self,
        logits: Float[np.ndarray, "flattened_batch_size seq_len ..."],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " batch_size"]:
        reshaped_logits = Postprocessors._reshape(logits, original_shape)
        # Calculate rolling mean across sequence length dimension
        batch_size, seq_len, embed_dim = reshaped_logits.shape

        # Only include complete windows
        n_windows = seq_len - self.window_size + 1
        if n_windows < 1:
            raise ValueError(
                f"Window size {self.window_size} is larger than sequence length {seq_len}"
            )

        # Create rolling windows and take mean
        rolling_means = np.zeros((batch_size, n_windows, embed_dim))
        for i in range(n_windows):
            window = reshaped_logits[:, i : i + self.window_size, :]
            rolling_means[:, i, :] = window.mean(axis=1)

        return Postprocessors.sigmoid(rolling_means.max(axis=1), original_shape)

    @property
    def id(self) -> str:
        return f"max_of_rolling_mean_{self.window_size}"


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
