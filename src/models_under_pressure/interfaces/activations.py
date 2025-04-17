from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import Dataset as TorchDataset

from models_under_pressure.interfaces.dataset import BaseDataset


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
        # Convert to activations here:
        self._activations = torch.Tensor(activations.get_activations(per_token=True))
        self._attention_mask = torch.Tensor(
            activations.get_attention_mask(per_token=True)
        )
        self._input_ids = torch.Tensor(activations.get_input_ids(per_token=True))
        self.y = torch.Tensor(y)

        assert (
            len(self._activations.shape) == 2
        ), f"Activations must be 2D, got {self._activations.shape}"
        assert (
            len(self._attention_mask.shape) == 1
        ), f"Attention mask must be 1D, got {self._attention_mask.shape}"
        assert (
            len(self._input_ids.shape) == 1
        ), f"Input ids must be 1D, got {self._input_ids.shape}"

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
        """
        Returns the masked activations, attention mask, input ids and labels.
        """

        # Multiply the attention mask to the activations:
        acts = self.activations[index] * self.attention_mask[index]

        return (
            acts,  # masked activations
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
        self._activations = torch.Tensor(activations._activations)
        self._attention_mask = torch.Tensor(activations._attention_mask)
        self._input_ids = torch.Tensor(activations._input_ids)
        self.y = torch.Tensor(y)

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
        """
        Return the masked activations, attention mask, and input ids.

        TODO: write this to take a slice?
        """

        acts = self.activations[index] * self.attention_mask[index][:, None]

        return (
            acts,  # masked activations
            self._attention_mask[index],
            self._input_ids[index],
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
        input_ids = einops.rearrange(self._input_ids.numpy(), "b s -> (b s)")

        return ActivationPerTokenDataset(
            activations=Activation(activations, attention_mask, input_ids), y=y
        )


@dataclass
class Activation:
    _activations: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    _attention_mask: Float[torch.Tensor, "batch_size seq_len"]
    _input_ids: Float[torch.Tensor, "batch_size seq_len"]

    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> "Activation":
        return cls(
            _activations=dataset.other_fields["activations"],  # type: ignore
            _attention_mask=dataset.other_fields["attention_mask"],  # type: ignore
            _input_ids=dataset.other_fields["input_ids"],  # type: ignore
        )

    @classmethod
    def concatenate(cls, activations: list["Activation"]) -> "Activation":
        return Activation(
            _activations=torch.cat([a._activations for a in activations], dim=0),
            _attention_mask=torch.cat([a._attention_mask for a in activations], dim=0),
            _input_ids=torch.cat([a._input_ids for a in activations], dim=0),
        )

    def get_activations(
        self, per_token: bool = False
    ) -> Float[np.ndarray, "batch_size seq_len embed_dim"]:
        activations = self._activations.float().numpy()
        if per_token:
            return einops.rearrange(activations, "b s e -> (b s) e")
        else:
            return activations

    def get_attention_mask(
        self, per_token: bool = False
    ) -> Float[np.ndarray, "batch_size seq_len"]:
        attention_mask = self._attention_mask.numpy()
        if per_token:
            return einops.rearrange(attention_mask, "b s -> (b s)")
        else:
            return attention_mask

    def get_input_ids(
        self, per_token: bool = False
    ) -> Float[np.ndarray, " batch_size"]:
        input_ids = self._input_ids.numpy()
        if per_token:
            return einops.rearrange(input_ids, "b s -> (b s)")
        else:
            return input_ids

    def __post_init__(self):
        """Validate shapes after initialization, save the input shapes for later use."""
        batch_size, seq_len, embed_dim = self._activations.shape

        # Save the data shapes for easy access
        self.batch_size: int = batch_size
        self.seq_len: int = seq_len
        self.embed_dim: int = embed_dim

        assert self._attention_mask.shape == (batch_size, seq_len), (
            f"Attention mask shape {self._attention_mask.shape} does not match "
            f"expected shape ({batch_size}, {seq_len})"
        )
        assert self._input_ids.shape == (batch_size, seq_len), (
            f"Input ids shape {self._input_ids.shape} does not match "
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
        activations_split = torch.split(self._activations, indices)
        attention_mask_split = torch.split(self._attention_mask, indices)
        input_ids_split = torch.split(self._input_ids, indices)

        return [
            Activation(act, mask, ids)
            for act, mask, ids in zip(
                activations_split, attention_mask_split, input_ids_split
            )
        ]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._activations.shape

    def to_dataset(
        self, y: Float[torch.Tensor, " batch_size"], per_token: bool = False
    ) -> "ActivationDataset | ActivationPerTokenDataset":
        if per_token:
            # Repeat y to match flattened sequence length
            y = einops.repeat(y, "b -> (b s)", s=self.seq_len)
            return ActivationPerTokenDataset(activations=self, y=y)
        else:
            return ActivationDataset(activations=self, y=y)


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
