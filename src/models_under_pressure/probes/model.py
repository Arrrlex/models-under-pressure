from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)

from models_under_pressure.config import DEVICE
from models_under_pressure.interfaces.dataset import Dialogue


@dataclass
class LLMModel:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase

    @classmethod
    def load(cls, model_name: str, tokenizer_name: str | None = None) -> "LLMModel":
        if tokenizer_name is None:
            tokenizer_name = model_name

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return cls(model, tokenizer)

    def tokenize(
        self,
        dialogues: list[Dialogue],
        add_generation_prompt: bool = True,
        **tokenizer_kwargs: Any,
    ) -> list[list[int]]:
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

    def get_activations(
        self, dialogues: list[Dialogue], layers: list[int] | None = None
    ) -> torch.Tensor:
        if layers is None:
            layers = list(range(self.model.config.n_layers))

        inputs = self.tokenize(dialogues)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Dictionary to store residual activations
        activations = []

        # Hook function to capture residual activations before layernorm
        def hook_fn(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            activations.append(input[0].detach().cpu())  # Store the residual connection

        # Register hooks on each transformer block (LLaMA layers)
        hooks: list[torch.nn.Module] = [
            # Pre-attention residual
            layer.input_layernorm.register_forward_hook(hook_fn)
            for layer in model.model.layers
        ]

        # Forward pass
        _ = model(**inputs)

        # Remove hooks after capturing activations
        for hook in hooks:
            hook.remove()

        # Print stored activations
        for i, act in enumerate(activations):
            print(f"Layer: {i}, Activation Shape: {act.shape}")

        all_acts = torch.stack(activations)
        print("All activations shape:", all_acts.shape)

        return all_acts.cpu().detach().numpy()
