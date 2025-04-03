from typing import Any, Callable, Self, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models_under_pressure.config import BATCH_SIZE, CACHE_DIR, DEVICE, MODEL_MAX_MEMORY
from models_under_pressure.interfaces.dataset import BaseDataset, Input, to_dialogue
from models_under_pressure.utils import batched_range, hf_login


class HookedModel:
    def __init__(self, model: torch.nn.Module, layers: list[int]):
        self.model = model
        self.layers = layers
        self.cache = {}
        self.hooks = []

    def make_hook(self, layer: int) -> Callable:
        def hook_fn(module, input, output):  # type: ignore
            self.cache[layer] = output.cpu()

        return hook_fn

    def __enter__(self) -> Self:
        hook_fns = [self.make_hook(layer) for layer in self.layers]

        # Different model architectures have different structures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # LLaMA-style architecture
            for layer, hook_fn in zip(self.layers, hook_fns):
                resid = self.model.model.layers[layer].input_layernorm
                self.hooks.append(resid.register_forward_hook(hook_fn))

        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            # GPT-style architecture (like Qwen)
            for layer, hook_fn in zip(self.layers, hook_fns):
                resid = self.model.transformer.h[layer].ln_1
                self.hooks.append(resid.register_forward_hook(hook_fn))
        else:
            raise ValueError(
                f"Unsupported model architecture: {type(self.model)}. "
                "Cannot locate transformer layers."
            )

        return self

    def get_acts(self, batch_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        _ = self.model(**batch_inputs)
        activations = [self.cache[layer] for layer in self.layers]
        return torch.stack(activations, dim=0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()


class LLMModel:
    def __init__(
        self,
        model_name: str,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE,
        tokenize_kwargs: dict[str, Any] | None = None,
        **model_tokenizer_kwargs: Any,
    ):
        hf_login()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        kwargs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": device,
            "torch_dtype": dtype,
            "cache_dir": CACHE_DIR,
            "max_memory": MODEL_MAX_MEMORY.get(model_name),
            **model_tokenizer_kwargs,
        }
        self.model = AutoModelForCausalLM.from_pretrained(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(**kwargs)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        default_tokenize_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": True,
            "max_length": 2**13,
        }

        self.tokenize_kwargs = default_tokenize_kwargs | (tokenize_kwargs or {})

        self.name = model_name
        self.batch_size = batch_size
        self.device = device
        self._cache = {}
        self._hooks = []

    @property
    def n_layers(self) -> int:
        # Use num_hidden_layers for LLaMA models, otherwise n_layers
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers  # type: ignore
        elif hasattr(self.model.config, "n_layers"):
            return self.model.config.n_layers  # type: ignore
        else:
            raise ValueError(
                f"Model {self.model.name_or_path} has no num_hidden_layers or n_layers attribute"
            )

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)

    def tokenize(self, dialogues: Sequence[Input]) -> dict[str, torch.Tensor]:
        dialogues = [to_dialogue(d) for d in dialogues]
        input_dicts = [[d.model_dump() for d in dialogue] for dialogue in dialogues]

        input_str = self.tokenizer.apply_chat_template(
            input_dicts,
            tokenize=False,  # Return string instead of tokens
            add_generation_prompt=False,  # Add final assistant prefix for generation
        )

        token_dict = self.tokenizer(input_str, **self.tokenize_kwargs)  # type: ignore
        for k, v in token_dict.items():
            if k in ["input_ids", "attention_mask"]:
                token_dict[k] = v[:, 1:]
            if isinstance(v, torch.Tensor):
                token_dict[k] = v.to(next(self.model.parameters()).device)

        # Check that attention mask exists in token dict
        if "attention_mask" not in token_dict:
            raise ValueError("Tokenizer output must include attention mask")

        return token_dict  # type: ignore

    @torch.no_grad()
    def get_batched_activations(
        self,
        dataset: BaseDataset,
        layers: list[int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Get activations for a given model and config.

        Handles batching of activations.
        """

        hidden_dim = self.model.config.hidden_size  # type: ignore
        n_samples = len(dataset.inputs)

        # Tokenize entire dataset at once
        inputs = self.tokenize(dataset.inputs)
        n_samples, max_seq_len = inputs["input_ids"].shape

        # Create empty tensor for all activations
        all_activations = torch.empty(
            (len(layers), n_samples, max_seq_len, hidden_dim),
            device="cpu",
            dtype=torch.float16,
        )

        with HookedModel(self.model, layers) as hooked_model:
            # Process each batch
            for start_idx, end_idx in tqdm(
                batched_range(n_samples, self.batch_size),
                desc="Generating activations...",
            ):
                # Get batch of tokenized inputs
                batch_inputs = {k: v[start_idx:end_idx] for k, v in inputs.items()}

                # Get activations for this batch
                activations = hooked_model.get_acts(batch_inputs)

                # Write to the relevant slice of the big tensor
                all_activations[:, start_idx:end_idx] = activations

        return all_activations, inputs
