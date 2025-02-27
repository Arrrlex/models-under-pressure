from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)

from models_under_pressure.config import DEVICE
from models_under_pressure.interfaces.dataset import Dialogue, Input, Message


@dataclass
class LLMModel:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase

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

        return cls(model, tokenizer)

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
        inputs: list[Input],
        layers: list[int] | None = None,
    ) -> torch.Tensor:  # TODO Also return attention mask
        dialogues = [
            Dialogue([Message(role="user", content=inp)])
            if isinstance(inp, str)
            else inp  # Dialogue type
            for inp in inputs
        ]

        if layers is None:
            # Use num_hidden_layers for LLaMA models, otherwise n_layers
            if hasattr(self.model.config, "num_hidden_layers"):
                layers = list(range(self.model.config.num_hidden_layers))
            else:
                layers = list(range(self.model.config.n_layers))

        torch_inputs = self.tokenize(dialogues)  # type: ignore

        # Dictionary to store residual activations
        activations = []

        # Hook function to capture residual activations before layernorm
        def hook_fn(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            activations.append(input[0].detach().cpu())  # Store the residual connection

        # Register hooks on each transformer block based on model architecture
        hooks = []

        # Different model architectures have different structures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # LLaMA-style architecture
            for layer in self.model.model.layers:
                hooks.append(layer.input_layernorm.register_forward_hook(hook_fn))
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            # GPT-style architecture (like Qwen)
            for layer in self.model.transformer.h:
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

        # Print stored activations
        for i, act in enumerate(activations):
            print(f"Layer: {i}, Activation Shape: {act.shape}")

        all_acts = torch.stack(activations)
        print("All activations shape:", all_acts.shape)

        return all_acts.cpu().detach().numpy()


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
        Dialogue(
            [
                Message(
                    role="user",
                    content="Hello, how are you?",
                )
            ]
        ),
        Dialogue(
            [
                Message(
                    role="user",
                    content="Hello again!",
                )
            ]
        ),
    ]  # type: ignore
    activations = model.get_activations(inputs=dialogues)
    print(activations.shape)
