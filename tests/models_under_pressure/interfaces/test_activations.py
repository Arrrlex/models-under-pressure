import numpy as np

from models_under_pressure.interfaces.activations import Activation, Preprocessors


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

    assert np.allclose(result.activations, expected_activations)
    assert np.allclose(result.attention_mask, expected_attention_mask)


def test_mean_aggregation_random():
    """Test mean_aggregation with random data."""
    # Create random test data
    batch_size, seq_len, embed_dim = 2, 3, 6
    activations = np.random.randn(batch_size, seq_len, embed_dim)
    attention_mask = np.ones((batch_size, seq_len))
    input_ids = np.ones((batch_size, seq_len))

    # Create activation object and compute mean aggregation
    activation = Activation(activations, attention_mask, input_ids)
    result = activation.mean_aggregation()

    # Verify shapes and that mean was computed correctly
    assert result.activations.shape == (batch_size, embed_dim)
    assert result.attention_mask.shape == (batch_size, 1)

    # Manually compute expected result
    expected = activations.mean(axis=1)
    assert np.allclose(result.activations, expected)


def test_per_token_preprocessor():
    """
    Test per_token aggregator with a simple 4x3x2 activation array
    """
    # Create a 4x3x2 activation array
    batch_size, seq_len, embed_dim = 3, 2, 1
    acts = np.arange(batch_size * seq_len * embed_dim).reshape(
        batch_size, seq_len, embed_dim
    )

    attention_mask = np.ones((batch_size, seq_len))
    input_ids = np.ones((batch_size, seq_len))

    X = Activation(
        activations=acts,
        attention_mask=attention_mask,
        input_ids=input_ids,
    )

    y = np.array([0, 1, 2])

    X_new, y_new = Preprocessors.per_token(X, y)

    assert y_new is not None
    assert np.allclose(y_new, np.array([0, 0, 1, 1, 2, 2]))
    assert np.allclose(X_new, acts.reshape(-1, embed_dim))
