from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import Dataset as TorchDataset


class ActivationPerTokenDataset(TorchDataset):
    """
    A pytorch Dataset class that contains the activations structured as a flattened per-token dataset.
    Each activation and attention mask is batch_size * seq_len long.
    """

    def __init__(
        self,
        activations: "Activation",
        y: Float[np.ndarray, " batch_size"],
    ):
        self._activations = torch.Tensor(activations.activations)
        self._attention_mask = torch.Tensor(activations.attention_mask)
        self._input_ids = torch.Tensor(activations.input_ids)
        self.y = torch.Tensor(y)
        self.per_token: bool = True

    @property
    def activations(self) -> Float[torch.Tensor, "batch_size seq_len embed_dim"]:
        return self._activations

    @property
    def attention_mask(self) -> Float[torch.Tensor, "batch_size seq_len"]:
        return self._attention_mask

    @property
    def input_ids(self) -> Float[torch.Tensor, "batch_size seq_len"]:
        return self._input_ids

    def assert_per_token_dims(self):
        """
        Ensure the inputed data actually is per-token data. The shape of the tensors should be
        correct etc...
        """

        assert (
            self.activations.shape[0]
            == self.attention_mask.shape[0]
            == self.input_ids.shape[0]
            == self.y.shape[0]
        ), f"All tensors must have the same batch size,\
            got {self.activations.shape[0]},\
                {self.attention_mask.shape[0]},\
                {self.input_ids.shape[0]},\
                {self.y.shape[0]}"

        assert (
            len(self.activations.shape) == 2
        ), f"Activations must be 2D, got {self.activations.shape}"
        assert (
            len(self.attention_mask.shape) == 1
        ), f"Attention mask must be 1D, got {self.attention_mask.shape}"
        assert (
            len(self.input_ids.shape) == 1
        ), f"Input ids must be 1D, got {self.input_ids.shape}"
        assert len(self.y.shape) == 1, f"Labels must be 1D, got {self.y.shape}"

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.activations[index],
            self.attention_mask[index],
            self.input_ids[index],
            self.y[index],
        )


class ActivationDataset(TorchDataset):
    """
    A pytorch Dataset class that contains the activations structured as a batch-wise dataset.
    Each activation and attention mask is batch_size, seq_len, (embed_dim). The class contains
    methods to convert the dataset to a per-token dataset.
    """

    def __init__(
        self,
        activations: "Activation",
        y: Float[np.ndarray, " batch_size"],
    ):
        self._activations = torch.Tensor(activations.activations)
        self._attention_mask = torch.Tensor(activations.attention_mask)
        self.input_ids = torch.Tensor(activations.input_ids)
        self.y = torch.Tensor(y)
        self.per_token: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        batch_size, seq_len, embed_dim = self._activations.shape
        return batch_size, seq_len, embed_dim

    @property
    def attention_mask(self) -> Float[torch.Tensor, "batch_size seq_len"]:
        return self._attention_mask

    @property
    def activations(self) -> Float[torch.Tensor, "batch_size seq_len embed_dim"]:
        return self._activations

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[
        Float[torch.Tensor, "batch_size seq_len embed_dim"],
        Float[torch.Tensor, "batch_size seq_len"],
        Float[torch.Tensor, "batch_size seq_len"],
        Float[torch.Tensor, " batch_size"],
    ]:
        return (
            self.activations[index],
            self.attention_mask[index],
            self.input_ids[index],
            self.y[index],
        )

    def to_per_token(self) -> "ActivationPerTokenDataset":
        """
        Convert the dataset from a batch-wise dataset to a flattened per-token dataset.
        """

        # Get the shape of the activations
        _, seq_len, _ = self.activations.shape

        # Shape: (batch_size, seq_len, embed_dim)
        # masked_acts = self.activations * self.attention_mask[:, :, None]

        # Repeat y to match flattened sequence length
        y = einops.repeat(self.y.numpy(), "b -> (b s)", s=seq_len)

        # Shape: (batch_size * seq_len, embed_dim)
        activations = einops.rearrange(self.activations.numpy(), "b s e -> (b s) e")
        attention_mask = einops.rearrange(self.attention_mask.numpy(), "b s -> (b s)")
        input_ids = einops.rearrange(self.input_ids.numpy(), "b s -> (b s)")

        return ActivationPerTokenDataset(
            activations=Activation(activations, attention_mask, input_ids), y=y
        )


@dataclass
class Activation:
    activations: Float[np.ndarray, "batch_size seq_len embed_dim"]
    attention_mask: Float[np.ndarray, "batch_size seq_len"]
    input_ids: Float[np.ndarray, "batch_size seq_len"]
    per_token: bool = False

    @classmethod
    def concatenate(cls, activations: list["Activation"]) -> "Activation":
        return Activation(
            activations=np.concatenate([a.activations for a in activations], axis=0),
            attention_mask=np.concatenate(
                [a.attention_mask for a in activations], axis=0
            ),
            input_ids=np.concatenate([a.input_ids for a in activations], axis=0),
        )

    def __post_init__(self):
        """Validate shapes after initialization, save the input shapes for later use."""
        batch_size, seq_len, embed_dim = self.activations.shape

        # Save the data shapes for easy access
        self.batch_size: int = batch_size
        self.seq_len: int = seq_len
        self.embed_dim: int = embed_dim

        assert self.attention_mask.shape == (batch_size, seq_len), (
            f"Attention mask shape {self.attention_mask.shape} does not match "
            f"expected shape ({batch_size}, {seq_len})"
        )
        assert self.input_ids.shape == (batch_size, seq_len), (
            f"Input ids shape {self.input_ids.shape} does not match "
            f"expected shape ({batch_size}, {seq_len})"
        )

    def split(self, indices: list[int]) -> list["Activation"]:
        """Split the activation into multiple parts based on the given indices.

        Args:
            indices: List of indices where to split the activation along the batch dimension.
                    For example, if indices=[2, 5], the result will have three parts:
                    [0:2], [2:5], and [5:end].

        Returns:
            List of Activation objects, each containing a portion of the original data.
        """
        activations_split = np.split(self.activations, indices)
        attention_mask_split = np.split(self.attention_mask, indices)
        input_ids_split = np.split(self.input_ids, indices)

        return [
            Activation(act, mask, ids)
            for act, mask, ids in zip(
                activations_split, attention_mask_split, input_ids_split
            )
        ]

    @property
    def shape(self) -> tuple[int, int, int]:
        batch_size, seq_len, embed_dim = self.activations.shape
        return batch_size, seq_len, embed_dim

    def to_dataset(
        self, y: Float[np.ndarray, " batch_size"]
    ) -> "ActivationDataset | ActivationPerTokenDataset":
        if self.per_token:
            return ActivationPerTokenDataset(activations=self, y=y)
        else:
            return ActivationDataset(activations=self, y=y)

    def to_per_token(self) -> "Activation":
        _, seq_len, _ = self.activations.shape
        assert self.per_token is False, "Already per-token"
        self.per_token = True
        return Activation(
            activations=einops.rearrange(self.activations, "b s e -> (b s) e"),
            attention_mask=einops.rearrange(self.attention_mask, "b s -> (b s)"),
            input_ids=einops.rearrange(self.input_ids, "b s -> (b s)"),
        )

    def to_per_entry(self) -> "Activation":
        """
        Convert the activation from a per-token dataset to a per-entry dataset.
        """
        _, seq_len, _ = self.activations.shape
        assert self.per_token is True, "Already per-entry"
        self.per_token = False

        # Shape: (batch_size * seq_len, embed_dim)
        activations = einops.rearrange(
            self.activations, "(b s) e -> b s e", b=self.batch_size, s=self.seq_len
        )
        attention_mask = einops.rearrange(
            self.attention_mask, "(b s) -> b s", b=self.batch_size, s=self.seq_len
        )
        input_ids = einops.rearrange(
            self.input_ids, "(b s) -> b s", b=self.batch_size, s=self.seq_len
        )

        return Activation(
            activations=activations,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )


class Preprocessor(Protocol):
    def __call__(
        self, X: Activation, y: Optional[Float[np.ndarray, " batch_size"]] = None
    ) -> Tuple[
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
        batch_size, seq_len, embed_dim = X.activations.shape
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
            window_acts = X.activations[:, start_idx:end_idx, :]
            # Get corresponding attention mask values
            window_mask = X.attention_mask[:, start_idx:end_idx, None]

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
        X: Activation, y: Optional[Float[np.ndarray, " batch_size"]] = None
    ) -> Tuple[
        Float[np.ndarray, " batch_size embed_dim"],
        Optional[Float[np.ndarray, " batch_size"]],
    ]:
        # Shape: (batch_size, seq_len, embed_dim)
        masked_acts = X.activations * X.attention_mask[:, :, None]

        # Shape: (batch_size, embed_dim)
        # Sum and divide by the number of non-masked tokens
        sum_acts = masked_acts.sum(axis=1)

        # Shape: (batch_size, 1)
        # Add small epsilon to avoid division by zero
        token_counts = X.attention_mask.sum(axis=1, keepdims=True) + 1e-10

        # Shape: (batch_size, embed_dim)
        # Calc mean of the number of non-masked tokens
        return sum_acts / token_counts, y

    @staticmethod
    def per_token(
        X: Activation, y: Optional[Float[np.ndarray, " batch_size"]] = None
    ) -> Tuple[
        Float[np.ndarray, "batch_size * seq_len embed_dim"],
        Optional[Float[np.ndarray, " batch_size * seq_len"]],
    ]:
        _, seq_len, _ = X.activations.shape

        # Shape: (batch_size, seq_len, embed_dim)
        masked_acts = X.activations * X.attention_mask[:, :, None]

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
    ) -> Tuple[
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
