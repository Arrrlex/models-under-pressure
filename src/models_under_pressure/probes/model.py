from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from jaxtyping import Float
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)

from models_under_pressure.config import DEVICE
from models_under_pressure.interfaces.dataset import (
    Dialogue,
    Input,
    Message,
    to_dialogue,
)


@dataclass
class LLMModel:
    name: str
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase

    @property
    def n_layers(self) -> int:
        # Use num_hidden_layers for LLaMA models, otherwise n_layers
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers
        else:
            return self.model.config.n_layers

    @classmethod
    def load(
        cls,
        model_name: str,
        tokenizer_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> "LLMModel":
        if tokenizer_name is None:
            tokenizer_name = model_name

        if model_kwargs is None:
            model_kwargs = {}

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return cls(name=model_name, model=model, tokenizer=tokenizer)

    def tokenize(
        self,
        dialogues: list[Dialogue],
        add_generation_prompt: bool = True,
        **tokenizer_kwargs: Any,
    ) -> torch.Tensor:
        default_tokenizer_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": True,
            "max_length": 1028,  # todo: later we will want to deal with very long sequences.....
        }

        tokenizer_kwargs = default_tokenizer_kwargs | tokenizer_kwargs

        return self.tokenizer.apply_chat_template(
            [[d.model_dump() for d in dialogue] for dialogue in dialogues],
            tokenize=True,  # Return string instead of tokens
            add_generation_prompt=add_generation_prompt,  # Add final assistant prefix for generation
            **tokenizer_kwargs,
        )  # type: ignore

    #
    @torch.no_grad()
    def get_activations(
        self,
        inputs: Sequence[Input],
        layers: Sequence[int] | None = None,
    ) -> Float[
        np.ndarray, "layers batch_size seq_len embed_dim"
    ]:  # TODO Also return attention mask
        # TODO Implement mini-batches
        dialogues = [to_dialogue(inp) for inp in inputs]
        layers = layers or list(range(self.n_layers))

        torch_inputs = self.tokenize(dialogues)  # type: ignore

        # Dictionary to store residual activations
        activations = []

        # Hook function to capture residual activations before layernorm
        def hook_fn(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            activations.append(
                input[0].detach().cpu().numpy()
            )  # Store the residual connection

        # Register hooks on each transformer block based on model architecture
        hooks = []

        # Different model architectures have different structures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # LLaMA-style architecture
            for i in layers:
                layer = self.model.model.layers[i]
                hooks.append(layer.input_layernorm.register_forward_hook(hook_fn))

        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            # GPT-style architecture (like Qwen)
            for i in layers:
                layer = self.model.transformer.h[i]
                hooks.append(layer.ln_1.register_forward_hook(hook_fn))
        else:
            raise ValueError(
                f"Unsupported model architecture: {type(self.model)}. "
                "Cannot locate transformer layers."
            )

        # Forward pass
        _ = self.model(torch_inputs.to(DEVICE))

        # Remove hooks after capturing activations
        for hook in hooks:
            hook.remove()

        assert (
            len(activations) == len(layers)
        ), f"Number of activations ({len(activations)}) does not match number of layers ({len(layers)})"

        # Print stored activations
        for layer, act in zip(layers, activations):
            print(f"Layer: {layer}, Activation Shape: {act.shape}")

        return np.stack(activations)


if __name__ == "__main__":
    import os

    import dotenv

    dotenv.load_dotenv()

    model = LLMModel.load(
        # "Qwen/Qwen2.5-0.5B-Instruct",
        "meta-llama/LLama-3.2-1B-Instruct",
        # model_kwargs={"trust_remote_code": True},
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        # tokenizer_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )
    print(model.model.config)
    dialogues = [
        [
            Message(
                role="user",
                content="Hello, how are you?",
            )
        ],
        [
            Message(
                role="user",
                content="Hello again!",
            )
        ],
    ]  # type: ignore
    activations = model.get_activations(inputs=dialogues)
    print(activations.shape)
