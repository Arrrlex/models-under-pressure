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


def test_mean_aggregation():
    """Test the mean_aggregation method with a simple example."""
    # Create a simple test case with 2 samples, 3 tokens each, and 2-dimensional embeddings
    # Sample 1: All tokens are valid
    # Sample 2: Only first two tokens are valid (third is masked)
    activations = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Sample 1
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],  # Sample 2
        ]
    )
    attention_mask = np.array(
        [
            [1, 1, 1],  # All tokens in sample 1 are valid
            [1, 1, 0],  # Only first two tokens in sample 2 are valid
        ]
    )
    input_ids = np.array([[101, 102, 103], [104, 105, 106]])

    activation = Activation(activations, attention_mask, input_ids)
    result = activation.mean_aggregation()

    # Expected results:
    # Sample 1: Mean of all three tokens = ([1,2] + [3,4] + [5,6])/3 = [3, 4]
    # Sample 2: Mean of first two tokens = ([7,8] + [9,10])/2 = [8, 9]
    expected_activations = np.array(
        [
            [3.0, 4.0],  # Mean of sample 1
            [8.0, 9.0],  # Mean of sample 2
        ]
    )
    expected_attention_mask = np.ones((2, 1))

    print("Test mean_aggregation:")
    print(f"Result activations shape: {result.activations.shape}")
    print(f"Expected: {expected_activations}")
    print(f"Got: {result.activations}")

    assert np.allclose(result.activations, expected_activations)
    assert np.allclose(result.attention_mask, expected_attention_mask)
    print("Mean aggregation test passed!")


if __name__ == "__main__":
    activations = np.random.randn(2, 3, 6)
    attention_mask = np.ones((2, 3))
    input_ids = np.ones((2, 3))
    print(Activation(activations, attention_mask, input_ids).mean_aggregation())

    # Run tests
    test_mean_aggregation()

    # list of input tokens and corresponding activations, and the predicted output
