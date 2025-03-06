from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from jaxtyping import Float


class AggregationType(Enum):
    MEAN = auto()
    ROLLING_MEAN = auto()
    NONE = auto()


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

    def rolling_mean_aggregation(
        self,
        window_size: int = 3,
    ) -> "Activation":
        """Compute rolling window mean of activations, respecting attention mask."""
        batch_size, seq_len, embed_dim = self.activations.shape
        result = np.zeros((batch_size, embed_dim))

        for i in range(batch_size):
            # Get valid token positions
            valid_indices = np.where(self.attention_mask[i] > 0)[0]
            if len(valid_indices) == 0:
                continue
            # For each valid position, compute mean of window centered at that position
            valid_acts = []
            for idx in valid_indices:
                start = max(0, idx - window_size // 2)
                end = min(seq_len, idx + window_size // 2 + 1)

                # Only consider valid tokens within window
                window_indices = [
                    j for j in range(start, end) if self.attention_mask[i, j] > 0
                ]
                if window_indices:
                    window_mean = self.activations[i, window_indices, :].mean(axis=0)
                    valid_acts.append(window_mean)

            if valid_acts:
                result[i] = np.mean(valid_acts, axis=0)

        attention_mask = np.ones(shape=(self.activations.shape[0], 1))
        return Activation(result, attention_mask, self.input_ids)
