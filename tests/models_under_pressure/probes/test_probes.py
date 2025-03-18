from unittest.mock import Mock

import numpy as np
import pytest

from models_under_pressure.interfaces.activations import (
    Activation,
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import (
    Dataset,
    Label,
    LabelledDataset,
)
from models_under_pressure.probes.probes import LinearProbe, compute_accuracy


@pytest.fixture
def mock_llm():
    llm = Mock()
    # Mock get_batched_activations to return fake activation data
    llm.get_batched_activations.return_value = Activation(
        activations=np.random.randn(2, 3, 4),  # (batch_size, seq_len, hidden_dim)
        attention_mask=np.ones((2, 3)),  # (batch_size, seq_len)
        input_ids=np.ones((2, 3), dtype=np.int64),  # (batch_size, seq_len)
    )
    return llm


@pytest.fixture
def mock_dataset():
    return LabelledDataset(
        inputs=["text1", "text2"],
        ids=["1", "2"],
        other_fields={"labels": [0, 1]},
    )


@pytest.fixture
def aggregator():
    return Aggregator(
        preprocessor=Preprocessors.mean,
        postprocessor=Postprocessors.sigmoid,
    )


def test_linear_probe_fit(
    mock_llm: Mock,
    mock_dataset: LabelledDataset,
    aggregator: Aggregator,
):
    probe = LinearProbe(_llm=mock_llm, layer=0, aggregator=aggregator)
    probe.fit(mock_dataset)

    # Verify LLM was called correctly
    mock_llm.get_batched_activations.assert_called_once_with(
        dataset=mock_dataset,
        layer=0,
    )


def test_linear_probe_predict(mock_llm: Mock, mock_dataset: LabelledDataset):
    dataset = Dataset(
        inputs=["text1", "text2"],
        ids=["1", "2"],
        other_fields={},
    )
    aggregator = Aggregator(
        preprocessor=Preprocessors.mean,  # Use mean instead of per_token to get 2D array
        postprocessor=Postprocessors.sigmoid,
    )
    probe = LinearProbe(_llm=mock_llm, layer=0, aggregator=aggregator)

    # First fit the probe with some dummy data
    probe._fit(
        activations=Activation(
            activations=np.random.randn(2, 3, 4),
            attention_mask=np.ones((2, 3)),
            input_ids=np.ones((2, 3), dtype=np.int64),
        ),
        y=np.array([0, 1]),
    )

    predictions = probe.predict(dataset)
    assert len(predictions) == 2
    assert all(isinstance(pred, Label) for pred in predictions)


def test_per_token_predictions(mock_llm: Mock, aggregator: Aggregator):
    probe = LinearProbe(_llm=mock_llm, layer=0, aggregator=aggregator)

    # First fit the probe with some dummy data
    probe._fit(
        activations=Activation(
            activations=np.random.randn(2, 3, 4),
            attention_mask=np.ones((2, 3)),
            input_ids=np.ones((2, 3), dtype=np.int64),
        ),
        y=np.array([0, 1]),
    )

    inputs = ["text1", "text2"]
    predictions = probe.per_token_predictions(inputs)

    assert predictions.shape == (2, 3)  # (batch_size, seq_len)
    assert np.all((predictions == -1) | ((predictions >= 0) & (predictions <= 1)))


def test_compute_accuracy(aggregator: Aggregator):
    probe = LinearProbe(_llm=Mock(), layer=0, aggregator=aggregator)
    dataset = LabelledDataset(
        inputs=["text1", "text2"],
        ids=["1", "2"],
        other_fields={"labels": [0, 1]},
    )
    activations = Activation(
        activations=np.random.randn(2, 3, 4),
        attention_mask=np.ones((2, 3)),
        input_ids=np.ones((2, 3), dtype=np.int64),
    )

    # Mock the probe's predict method to return known values
    probe._predict = Mock(return_value=np.array([0, 1]))

    accuracy = compute_accuracy(probe, dataset, activations)
    assert (
        accuracy == 1.0
    )  # Should be perfect since we mocked predictions to match labels
