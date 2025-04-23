import numpy as np
import torch

from models_under_pressure.interfaces.activations import Activation, Preprocessors


def test_mean_preprocessor():
    activations = Activation(
        activations=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32
        ),  # (2, 2, 2)
        attention_mask=torch.ones((2, 2)),
        input_ids=torch.ones((2, 2), dtype=torch.int64),
    )

    processed, y = Preprocessors.mean(activations)
    assert y is None
    assert np.allclose(processed, [[[2.0, 3.0], [6.0, 7.0]]])
