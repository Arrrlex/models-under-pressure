from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jaxtyping import Float


@dataclass
class Activation:
    activations: Float[np.ndarray, "batch_size seq_len embed_dim"]
    attention_mask: Float[np.ndarray, "batch_size seq_len"]
    input_ids: Float[np.ndarray, "batch_size seq_len"]

    def mean_aggregation(self) -> "Activation":
        """Compute mean of activations across sequence length, respecting attention mask."""
        # Mask out padding tokens
        masked_acts = self.activations * self.attention_mask[:, :, None]
        # Sum and divide by the number of non-masked tokens
        sum_acts = masked_acts.sum(axis=1)
        # Add small epsilon to avoid division by zero
        token_counts = self.attention_mask.sum(axis=1, keepdims=True) + 1e-10
        # activations shape: (batch_size, 1, embed_dim)
        # attention_mask shape: (batch_size, 1)
        attention_mask = np.ones(shape=(self.activations.shape[0], 1))
        return Activation(sum_acts / token_counts, attention_mask, self.input_ids)


class Preprocessor(Protocol):
    def __call__(self, X: Activation) -> Float[np.ndarray, "batch_size ..."]: ...


class Postprocessor(Protocol):
    def __call__(
        self,
        logits: Float[np.ndarray, "flattened_batch_size ..."],
        original_shape: tuple[int, int, int],
    ) -> Float[np.ndarray, " batch_size"]: ...


class RollingMean(Preprocessor):
    window_size: int

    def __call__(
        self, X: Activation
    ) -> Float[np.ndarray, "batch_size shortened_seq_len embed_dim"]:
        batch_size, seq_len, embed_dim = X.activations.shape
        shortened_seq_len = seq_len - self.window_size + 1

        # If sequence is shorter than window_size, return empty array
        if shortened_seq_len <= 0:
            return np.zeros((batch_size, 0, embed_dim))

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

        return result

    @property
    def id(self) -> str:
        return f"rolling_mean_{self.window_size}"


class Preprocessors:
    @staticmethod
    def mean(X: Activation) -> Float[np.ndarray, "batch_size embed_dim"]:
        masked_acts = X.activations * X.attention_mask[:, :, None]
        # Sum and divide by the number of non-masked tokens
        sum_acts = masked_acts.sum(axis=1)
        # Add small epsilon to avoid division by zero
        token_counts = X.attention_mask.sum(axis=1, keepdims=True) + 1e-10
        # activations shape: (batch_size, 1, embed_dim)
        # attention_mask shape: (batch_size, 1)
        return sum_acts / token_counts

    @staticmethod
    def per_token(X: Activation) -> Float[np.ndarray, "batch_size seq_len embed_dim"]:
        return X.activations * X.attention_mask[:, :, None]


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

    def preprocess(self, X: Activation) -> Float[np.ndarray, "batch_size ..."]:
        self.original_shape = X.activations.shape  # type: ignore
        return self.preprocessor(X)

    def postprocess(
        self, logits: Float[np.ndarray, "flattened_batch_size ..."]
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
