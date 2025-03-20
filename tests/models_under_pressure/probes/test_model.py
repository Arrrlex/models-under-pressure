from typing import Callable
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerBase

from models_under_pressure.interfaces.dataset import Message
from models_under_pressure.probes.model import LLMModel


@pytest.fixture
def mock_model():
    model = Mock()
    # Mock the config attributes
    model.config = Mock()
    model.config.num_hidden_layers = 12

    # Create a mock parameter with 'cpu' device
    mock_param = Mock()
    mock_param.device = "cpu"

    # Create a function that returns a fresh list each time
    def get_parameters():
        return iter([mock_param])

    # Use this function for the parameters method
    model.parameters = get_parameters

    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "right"

    # Mock the chat template application
    tokenizer.apply_chat_template.return_value = "mocked chat template"

    # Mock the tokenization
    tokenizer.return_value = {
        "input_ids": torch.ones(2, 10),  # batch_size=2, seq_len=10
        "attention_mask": torch.ones(2, 10),
    }

    return tokenizer


@pytest.fixture
def llm_model(mock_model: Mock, mock_tokenizer: PreTrainedTokenizerBase):
    return LLMModel(
        name="test_model", model=mock_model, tokenizer=mock_tokenizer, device="cpu"
    )


def test_n_layers_property(llm_model: LLMModel):
    assert llm_model.n_layers == 12

    # Test fallback to n_layer
    delattr(llm_model.model.config, "num_hidden_layers")
    llm_model.model.config.n_layers = 24
    assert llm_model.n_layers == 24


def test_device_setter(llm_model: LLMModel):
    # Reset the mock to clear the call from initialization
    llm_model.model.to.reset_mock()

    # Now test setting the device
    llm_model.to("cuda")
    assert llm_model.device == "cuda"
    llm_model.model.to.assert_called_once_with("cuda")


@patch("models_under_pressure.probes.model.AutoModelForCausalLM")
@patch("models_under_pressure.probes.model.AutoTokenizer")
@patch("models_under_pressure.probes.model.hf_login")
def test_load_model(
    mock_hf_login: Mock, mock_auto_tokenizer: Mock, mock_auto_model: Mock
):
    mock_auto_model.from_pretrained.return_value = Mock()
    mock_auto_tokenizer.from_pretrained.return_value = Mock(
        pad_token_id=None, eos_token_id=2
    )

    model = LLMModel.load("test/model")

    assert isinstance(model, LLMModel)
    mock_auto_model.from_pretrained.assert_called_once()
    mock_auto_tokenizer.from_pretrained.assert_called_once()
    mock_hf_login.assert_called_once()


def test_tokenize(llm_model: LLMModel):
    dialogues = [[Message(role="user", content="Hello")]]

    # Mock the chat template to return a string
    llm_model.tokenizer.apply_chat_template.return_value = "mocked chat template"

    result = llm_model.tokenize(dialogues)  # type: ignore

    assert "input_ids" in result
    assert "attention_mask" in result
    llm_model.tokenizer.apply_chat_template.assert_called_once_with(
        [[{"role": "user", "content": "Hello"}]],  # Messages are converted to dicts
        tokenize=False,
        add_generation_prompt=True,
    )


def test_get_activations(llm_model: LLMModel):
    # Mock the layer structure for LLaMA-style architecture
    llm_model.model.model = Mock()
    llm_model.model.model.layers = [Mock() for _ in range(12)]

    # Create a mock layernorm with a patchable register_forward_hook
    mock_layernorm = Mock()
    # Using sequence length 9 since tokenize() removes the first token (BOS token)
    mock_activation = torch.ones(2, 9, 512)

    def mock_register_hook(hook_fn: Callable):  # type: ignore
        print("Inside mock_register_hook")
        hook_fn(None, [mock_activation], None)  # Note: activation should be in a list
        print("After hook_fn call")
        return Mock()

    mock_layernorm.register_forward_hook = mock_register_hook

    # Attach the mock layernorm to the layer
    for layer in llm_model.model.model.layers:
        layer.input_layernorm = mock_layernorm

    # Create test inputs
    inputs = [[Message(role="user", content="test")]]

    activation_obj = llm_model.get_activations(inputs, layer=0)

    assert activation_obj.get_activations().shape == (2, 9, 512)
    assert activation_obj.get_attention_mask().shape == (2, 9)
    assert activation_obj.get_input_ids().shape == (2, 9)


def test_get_batched_activations(llm_model: LLMModel):
    # Mock dataset with list of messages
    mock_dataset = Mock()
    mock_dataset.inputs = [[Message(role="user", content="test")] for _ in range(3)]

    # Mock get_activations to return consistent shapes
    def mock_get_activations(inputs: list, layer: int):
        batch_size = len(inputs)
        mock_obj = Mock(
            activations=np.ones((batch_size, 10, 512)),
            attention_mask=np.ones((batch_size, 10)),
            input_ids=np.ones((batch_size, 10)),
            get_activations=lambda per_token=False: np.ones((batch_size, 10, 512)),
            get_attention_mask=lambda per_token=False: np.ones((batch_size, 10)),
            get_input_ids=lambda per_token=False: np.ones((batch_size, 10)),
        )

        return mock_obj

    llm_model.get_activations = mock_get_activations  # type: ignore

    result = llm_model.get_batched_activations(mock_dataset, layer=0, batch_size=2)

    assert isinstance(result.get_activations(), np.ndarray)
    assert isinstance(result.get_attention_mask(), np.ndarray)
    assert isinstance(result.get_input_ids(), np.ndarray)
    assert result.get_activations().shape[0] == 3  # Total number of samples
