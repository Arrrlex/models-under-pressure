import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)

from models_under_pressure.config import BATCH_SIZE, DEVICE
from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import (
    Dataset,
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

    @property
    def device(self) -> str:
        return self.model.device

    @device.setter
    def device(self, device: str | None = None) -> None:
        dev = device or DEVICE
        self.model.to(dev)

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

        default_model_kwargs = {
            "token": os.getenv("HUGGINGFACE_TOKEN"),
            "device_map": DEVICE,
            "torch_dtype": torch.bfloat16 if "cuda" in DEVICE else torch.float16,
        }

        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs = default_model_kwargs | model_kwargs

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
    ) -> dict[str, torch.Tensor]:
        default_tokenizer_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": True,
            "max_length": 2048,  # todo: later we will want to deal with very long sequences.....
        }

        tokenizer_kwargs = default_tokenizer_kwargs | tokenizer_kwargs

        input_str = self.tokenizer.apply_chat_template(
            [[d.model_dump() for d in dialogue] for dialogue in dialogues],
            tokenize=False,  # Return string instead of tokens
            add_generation_prompt=add_generation_prompt,  # Add final assistant prefix for generation
        )

        token_dict = self.tokenizer(input_str, **tokenizer_kwargs)  # type: ignore

        for k, v in token_dict.items():
            if isinstance(v, torch.Tensor):
                token_dict[k] = v.to(self.device)

        # Check that attention mask exists in token dict
        if "attention_mask" not in token_dict:
            raise ValueError("Tokenizer output must include attention mask")

        return token_dict  # type: ignore

    #
    @torch.no_grad()
    def get_activations(
        self,
        inputs: Sequence[Input],
        layers: Sequence[int] | None = None,
    ) -> Activation:
        dialogues = [to_dialogue(inp) for inp in inputs]
        layers = layers or list(range(self.n_layers))

        torch_inputs = self.tokenize(dialogues)  # type: ignore

        # Dictionary to store residual activations
        activations = []

        # Hook function to capture residual activations before layernorm
        def hook_fn(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            activations.append(
                input[0].float().detach().cpu().numpy()
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
        _ = self.model(**torch_inputs)

        # Remove hooks after capturing activations
        for hook in hooks:
            hook.remove()

        assert (
            len(activations) == len(layers)
        ), f"Number of activations ({len(activations)}) does not match number of layers ({len(layers)})"

        # Print stored activations
        for layer, act in zip(layers, activations):
            print(f"Layer: {layer}, Activation Shape: {act.shape}")

        attention_mask = torch_inputs["attention_mask"].detach().cpu().numpy()
        input_ids = torch_inputs["input_ids"].detach().cpu().numpy()

        return Activation(np.stack(activations), attention_mask, input_ids)

    def get_batched_activations(
        self,
        dataset: Dataset,
        layer: int,
        batch_size: int = BATCH_SIZE,
    ) -> Activation:
        """
        Get activations for a given model and config.

        Handle batching of activations.
        """

        print("Generating activations...")

        n_samples = len(dataset.inputs)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Get the shape from first batch to ensure consistency
        first_batch = dataset.inputs[0:1]
        activation_obj = self.get_activations(inputs=first_batch, layers=[layer])
        first_activation = activation_obj.activations[0]
        first_attn_mask = activation_obj.attention_mask
        input_ids = activation_obj.input_ids
        activation_shape = first_activation.shape[1:]  # Remove batch dimension
        attn_mask_shape = first_attn_mask.shape[1:]  # Remove batch dimension

        all_activations = []
        all_attention_masks = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_inputs = dataset.inputs[start_idx:end_idx]

            activation_obj = self.get_activations(inputs=batch_inputs, layers=[layer])
            batch_activations = activation_obj.activations[0]
            batch_attn_mask = activation_obj.attention_mask

            # Ensure all batches have the same shape by padding/truncating
            if batch_activations.shape[1:] != activation_shape:
                padded_activations = np.zeros(
                    (batch_activations.shape[0],) + activation_shape
                )
                min_length = min(batch_activations.shape[1], activation_shape[0])
                padded_activations[:, :min_length] = batch_activations[:, :min_length]
                batch_activations = padded_activations

            if batch_attn_mask.shape[1:] != attn_mask_shape:
                padded_attn_mask = np.zeros(
                    (batch_attn_mask.shape[0],) + attn_mask_shape
                )
                min_length = min(batch_attn_mask.shape[1], attn_mask_shape[0])
                padded_attn_mask[:, :min_length] = batch_attn_mask[:, :min_length]
                batch_attn_mask = padded_attn_mask

            all_activations.append(batch_activations)
            all_attention_masks.append(batch_attn_mask)

        activations = np.concatenate(all_activations, axis=0)
        attention_mask = np.concatenate(all_attention_masks, axis=0)

        activations_obj = Activation(activations, attention_mask, input_ids)

        return activations_obj


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
    activation_obj = model.get_activations(inputs=dialogues)
    print(activation_obj.activations.shape)
    print(activation_obj.attention_mask.shape)
    print(activation_obj.input_ids.shape)
