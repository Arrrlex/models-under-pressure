import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)

from models_under_pressure.config import BATCH_SIZE, DEVICE
from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
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
    verbose: bool = False
    device: str = DEVICE

    @property
    def n_layers(self) -> int:
        # Use num_hidden_layers for LLaMA models, otherwise n_layers
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, "n_layers"):
            return self.model.config.n_layers
        else:
            raise ValueError(
                f"Model {self.model.name_or_path} has no num_hidden_layers or n_layers attribute"
            )

    def __setattr__(self, key: str, value: Any) -> None:
        super().__setattr__(key, value)
        if key == "device":
            self.model.to(value)

    @classmethod
    def load(
        cls,
        model_name: str,
        tokenizer_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> "LLMModel":
        if tokenizer_name is None:
            tokenizer_name = model_name

        default_model_kwargs = {
            "token": os.getenv("HUGGINGFACE_TOKEN"),
            "device_map": device or DEVICE,
            "torch_dtype": torch.bfloat16 if "cuda" in DEVICE else torch.float16,
        }

        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs = default_model_kwargs | model_kwargs

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        print("Model kwargs:", model_kwargs)

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

                if k == "input_ids":
                    token_dict[k] = v[:, 1:].to(self.device)
                elif k == "attention_mask":
                    token_dict[k] = v[:, 1:].to(self.device)

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
            if self.verbose:
                print(f"Layer: {layer}, Activation Shape: {act.shape}")

        attention_mask = torch_inputs["attention_mask"].detach().cpu().numpy()
        input_ids = torch_inputs["input_ids"].detach().cpu().numpy()

        return Activation(np.stack(activations), attention_mask, input_ids)

    def get_batched_activations(
        self,
        dataset: BaseDataset,
        layer: int,
        batch_size: int = BATCH_SIZE,
    ) -> Activation:
        """
        Get activations for a given model and config.

        Handle batching of activations.
        """

        print(f"Batch size: {batch_size}")

        n_samples = len(dataset.inputs)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_activations = []
        all_attention_masks = []
        all_input_ids = []
        for i in tqdm(range(n_batches), desc="Generating activations per batch..."):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_inputs = dataset.inputs[start_idx:end_idx]

            activation_obj = self.get_activations(inputs=batch_inputs, layers=[layer])
            batch_activations = activation_obj.activations[0]
            batch_attn_mask = activation_obj.attention_mask
            batch_input_ids = activation_obj.input_ids

            assert (
                len(batch_activations.shape) == 3
            ), f"Expected 3 dim activations, got {batch_activations.shape}"

            assert (
                len(batch_attn_mask.shape) == 2
            ), f"Expected 2 dim attention mask, got {batch_attn_mask.shape}"

            all_input_ids.append(batch_input_ids)
            all_activations.append(batch_activations)
            all_attention_masks.append(batch_attn_mask)

        # Process all_activations to ensure they have the same shape:
        # Find the maximum shape across all activations
        max_shape = max(act.shape[1] for act in all_activations)

        for i, act in enumerate(all_activations):
            if act.shape[1] != max_shape:
                # Create padding for activations and attention masks:
                act_padding = np.zeros(
                    (act.shape[0], max_shape - act.shape[1], act.shape[2])
                )
                attn_padding = np.zeros((act.shape[0], max_shape - act.shape[1]))
                input_ids_padding = np.zeros((act.shape[0], max_shape - act.shape[1]))

                # Append the padding to each activation and attention mask element:
                if self.tokenizer.padding_side == "left":
                    all_activations[i] = np.concatenate([act_padding, act], axis=1)
                    all_attention_masks[i] = np.concatenate(
                        [attn_padding, all_attention_masks[i]], axis=1
                    )
                    all_input_ids[i] = np.concatenate(
                        [input_ids_padding, all_input_ids[i]], axis=1
                    )
                else:
                    all_activations[i] = np.concatenate([act, act_padding], axis=1)
                    all_attention_masks[i] = np.concatenate(
                        [all_attention_masks[i], attn_padding], axis=1
                    )
                    all_input_ids[i] = np.concatenate(
                        [all_input_ids[i], input_ids_padding], axis=1
                    )

        activations = np.concatenate(all_activations, axis=0)
        attention_mask = np.concatenate(all_attention_masks, axis=0)
        input_ids = np.concatenate(all_input_ids, axis=0)

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
