from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from jaxtyping import Float


class AggregationType(Enum):
    MEAN = auto()
    ROLLING_MEAN = auto()


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


# Create test activation object with sample data
def test_mean_aggregation():
    """Test the mean_aggregation function."""
    # Test case 1: Simple mean with no masking
    activations = np.array(
        [
            [[1, 1], [2, 2], [3, 3]],  # First sequence
            [[4, 4], [5, 5], [6, 6]],  # Second sequence
        ]
    )
    attention_mask = np.ones((2, 3))
    input_ids = np.ones((2, 3))
    activation = Activation(activations, attention_mask, input_ids)

    result = activation.mean_aggregation()
    expected_means = np.array(
        [
            [2, 2],  # Mean of sequence 1: (1+2+3)/3 = 2
            [5, 5],  # Mean of sequence 2: (4+5+6)/3 = 5
        ]
    )
    assert np.allclose(result.activations, expected_means)

    # Test case 2: Mean with masked tokens
    attention_mask = np.array(
        [
            [1, 1, 0],  # Third token masked in first sequence
            [1, 0, 0],  # Second and third tokens masked in second sequence
        ]
    )
    activation = Activation(activations, attention_mask, input_ids)

    result = activation.mean_aggregation()
    expected_means = np.array(
        [
            [1.5, 1.5],  # Mean of sequence 1: (1+2)/2 = 1.5
            [4, 4],  # Mean of sequence 2: Just the first token
        ]
    )
    assert np.allclose(result.activations, expected_means)
    print("Mean aggregation tests passed!")


def test_rolling_mean_aggregation():
    """Test the rolling_mean_aggregation function."""
    # Test case 1: Rolling mean with no masking
    activations = np.array(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4]],  # First sequence
            [[5, 5], [6, 6], [7, 7], [8, 8]],  # Second sequence
        ]
    )
    attention_mask = np.ones((2, 4))
    input_ids = np.ones((2, 4))
    activation = Activation(activations, attention_mask, input_ids)

    result = activation.rolling_mean_aggregation(window_size=3)
    # For window_size=3, each position considers itself and adjacent positions
    expected_means = np.array(
        [
            [2, 2],  # Mean of windows in sequence 1
            [6.5, 6.5],  # Mean of windows in sequence 2
        ]
    )
    assert np.allclose(result.activations, expected_means)

    # Test case 2: Rolling mean with masked tokens
    attention_mask = np.array(
        [
            [1, 1, 1, 0],  # Last token masked in first sequence
            [1, 0, 1, 1],  # Second token masked in second sequence
        ]
    )
    activation = Activation(activations, attention_mask, input_ids)

    result = activation.rolling_mean_aggregation(window_size=3)
    expected_means = np.array(
        [
            [2, 2],  # Mean of valid windows in sequence 1
            [6.5, 6.5],  # Mean of valid windows in sequence 2, skipping masked token
        ]
    )
    assert np.allclose(result, expected_means)
    print("Rolling mean aggregation tests passed!")


if __name__ == "__main__":
    activations = np.random.randn(2, 3, 6)
    attention_mask = np.ones((2, 3))
    input_ids = np.ones((2, 3))
    print(Activation(activations, attention_mask, input_ids).mean_aggregation())

    # Run tests
    test_mean_aggregation()
    test_rolling_mean_aggregation()

    # list of input tokens and corresponding activations, and the predicted output
