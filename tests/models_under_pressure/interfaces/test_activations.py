import numpy as np
import torch

from models_under_pressure.interfaces.activations import Activation, Preprocessors


def test_per_token_preprocessor():
    """
    Test per_token aggregator with a simple 4x3x2 activation array
    """
    batch_size, seq_len, embed_dim = 3, 2, 1
    acts = np.arange(batch_size * seq_len * embed_dim).reshape(
        batch_size, seq_len, embed_dim
    )

    attention_mask = np.ones((batch_size, seq_len))
    input_ids = np.ones((batch_size, seq_len))

    X = Activation(
        _activations=acts,
        _attention_mask=attention_mask,
        _input_ids=input_ids,
    )

    y = np.array([0, 1, 2])

    X_new, y_new = Preprocessors.per_token(X, y)

    assert y_new is not None
    assert np.allclose(y_new, np.array([0, 0, 1, 1, 2, 2]))
    assert np.allclose(X_new, acts.reshape(-1, embed_dim))


def test_mean_preprocessor():
    activations = Activation(
        _activations=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32
        ),  # (2, 2, 2)
        _attention_mask=torch.ones((2, 2)),
        _input_ids=torch.ones((2, 2), dtype=torch.int64),
    )

    processed, y = Preprocessors.mean(activations)
    assert y is None
    assert np.allclose(processed, [[[2.0, 3.0], [6.0, 7.0]]])
