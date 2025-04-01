import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Sequence

import torch
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models_under_pressure.config import (
    ALL_DATASETS,
    BATCH_SIZE,
    CACHE_DIR,
    DEVICE,
    LOCAL_MODELS,
    MODEL_MAX_MEMORY,
    ACTIVATIONS_DIR,
)
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Input,
    LabelledDataset,
    to_dialogue,
)
from models_under_pressure.utils import hf_login


class ActivationsSpec(BaseModel):
    model_name: str
    layer: int
    dataset_name: str

    @cached_property
    def id(self) -> str:
        return hashlib.sha1(str(self).encode()).hexdigest()[:8]


@dataclass
class Activations:
    """Simple container class for activation tensors."""

    activations: torch.Tensor  # Shape: (n_layers, batch_size, seq_len, hidden_dim)
    attention_mask: torch.Tensor  # Shape: (batch_size, seq_len)
    input_ids: torch.Tensor  # Shape: (batch_size, seq_len)


def batched_range(n_samples: int, batch_size: int) -> list[tuple[int, int]]:
    """Generate start and end indices for batches of size batch_size.

    Args:
        n_samples: Total number of samples to process
        batch_size: Size of each batch

    Returns:
        List of (start_idx, end_idx) tuples for each batch
    """
    n_batches = (n_samples + batch_size - 1) // batch_size
    return [
        (i * batch_size, min((i + 1) * batch_size, n_samples)) for i in range(n_batches)
    ]


class LLMModel:
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

    def _add_hooks(self, layers: list[int]):
        def make_hook(layer: int):
            def hook_fn(module, input, output):  # type: ignore
                self._cache[layer] = output.cpu()

            return hook_fn

        hooks = [make_hook(layer) for layer in layers]

        # Different model architectures have different structures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # LLaMA-style architecture
            for layer, hook in zip(layers, hooks):
                resid = self.model.model.layers[layer].input_layernorm
                self._hooks.append(resid.register_forward_hook(hook))

        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            # GPT-style architecture (like Qwen)
            for layer, hook in zip(layers, hooks):
                resid = self.model.transformer.h[layer].ln_1
                self._hooks.append(resid.register_forward_hook(hook))
        else:
            raise ValueError(
                f"Unsupported model architecture: {type(self.model)}. "
                "Cannot locate transformer layers."
            )

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

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

    def get_activations_for_batch(
        self, batch_inputs: dict[str, torch.Tensor], layers: list[int]
    ) -> torch.Tensor:
        _ = self.model(**batch_inputs)
        activations = [self._cache[layer] for layer in layers]
        return torch.stack(activations, dim=0)  # type: ignore

    @torch.no_grad()
    def get_batched_activations(
        self,
        dataset: BaseDataset,
        layers: list[int],
    ) -> Activations:
        """
        Get activations for a given model and config.

        Handle batching of activations.
        """
        self._add_hooks(layers)

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        hidden_dim = self.model.config.hidden_size  # type: ignore
        n_samples = len(dataset.inputs)

        # Tokenize entire dataset at once
        inputs = self.tokenize(dataset.inputs)
        n_samples, max_seq_len = inputs["input_ids"].shape

        # Create empty tensor for all activations
        all_activations = torch.empty(
            (len(layers), n_samples, max_seq_len, hidden_dim),
            device=device,
            dtype=dtype,
        )

        # Process each batch
        for start_idx, end_idx in tqdm(
            batched_range(n_samples, self.batch_size),
            desc="Generating activations per batch...",
        ):
            # Get batch of tokenized inputs
            batch_inputs = {k: v[start_idx:end_idx] for k, v in inputs.items()}

            # Get activations for this batch
            batch_activations = self.get_activations_for_batch(batch_inputs, layers)

            # Write to the relevant slice of the big tensor
            all_activations[:, start_idx:end_idx] = batch_activations
        self._remove_hooks()
        return Activations(
            activations=all_activations,
            attention_mask=inputs["attention_mask"],
            input_ids=inputs["input_ids"],
        )


@dataclass
class ActivationStore:
    activations_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.activations_dir / "manifest.json"

    @property
    def tensors_dir(self) -> Path:
        return self.activations_dir / "tensors"

    def _add_spec_to_manifest(self, spec: ActivationsSpec):
        manifest = json.loads(self.manifest_path.read_text())
        manifest[spec.id] = spec.model_dump()
        self.manifest_path.write_text(json.dumps(manifest))

    def _remove_spec_from_manifest(self, spec: ActivationsSpec):
        manifest = json.loads(self.manifest_path.read_text())
        del manifest[spec.id]
        self.manifest_path.write_text(json.dumps(manifest))

    def save(
        self,
        model_name: str,
        dataset_name: str,
        layers: list[int],
        activations: Activations,
    ):
        for layer_idx, layer in tqdm(
            list(enumerate(layers)), desc="Saving activations..."
        ):
            spec = ActivationsSpec(
                model_name=model_name, dataset_name=dataset_name, layer=layer
            )

            # Save layer-specific data
            layer_data = {
                "activations": activations.activations[layer_idx],
                "attention_mask": activations.attention_mask,
                "input_ids": activations.input_ids,
            }
            save_file(layer_data, self.tensors_dir / f"{spec.id}.safetensors")
            self._add_spec_to_manifest(spec)

    def load(self, spec: ActivationsSpec) -> Activations:
        layer_data = load_file(self.tensors_dir / f"{spec.id}.safetensors")
        return Activations(
            activations=layer_data["activations"],
            attention_mask=layer_data["attention_mask"],
            input_ids=layer_data["input_ids"],
        )

    def delete(self, spec: ActivationsSpec):
        (self.activations_dir / f"{spec.id}.safetensors").unlink()
        self._remove_spec_from_manifest(spec)


def compute_activations_and_save(
    model_name: str,
    dataset_name: str,
    layers: list[int],
    activations_dir: Path,
):
    model = LLMModel(model_name)
    dataset = get_dataset(dataset_name)
    print("Getting activations...")
    activations = model.get_batched_activations(dataset, layers)
    store = ActivationStore(activations_dir)
    store.save(model.name, dataset_name, layers, activations)


def get_dataset(dataset_name: str) -> LabelledDataset:
    full_dataset_path = ALL_DATASETS[dataset_name]
    return LabelledDataset.load_from(full_dataset_path)


if __name__ == "__main__":
    compute_activations_and_save(
        model_name=LOCAL_MODELS["llama-1b"],
        dataset_name="synthetic_25_03_25",
        layers=[5, 6, 7, 8],
        activations_dir=ACTIVATIONS_DIR,
    )
