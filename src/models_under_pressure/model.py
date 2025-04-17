"""
Core module for working with language models and extracting their activations.

This module provides tools for:
1. Loading and managing language models
2. Extracting model activations at specific layers
3. Computing log likelihoods and generating text
4. Handling different model architectures (LLaMA-style and GPT-style)

The module handles batching and memory management to efficiently process large datasets
while working with potentially very large language models.
"""

from dataclasses import dataclass
from typing import Any, Callable, Self, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from models_under_pressure.config import BATCH_SIZE, CACHE_DIR, DEVICE, MODEL_MAX_MEMORY
from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Dialogue,
    Input,
    to_dialogue,
)
from models_under_pressure.utils import batched_range, hf_login


class HookedModel:
    """
    Context manager for extracting activations from specific model layers.

    Automatically detects and hooks into layer normalization points in both
    LLaMA-style and GPT-style architectures. Ensures proper cleanup of hooks
    to prevent memory leaks.
    """

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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        for hook in self.hooks:
            hook.remove()


@dataclass
class LLMModel:
    """
    High-level interface for working with language models.

    Provides unified access to:
    - Model loading and management
    - Tokenization
    - Activation extraction
    - Log likelihood computation
    - Text generation

    Handles architecture differences, tokenization, and memory management.
    """

    name: str
    device: str
    batch_size: int
    tokenize_kwargs: dict[str, Any]
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase

    @classmethod
    def load(
        cls,
        model_name: str,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE,
        tokenize_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """
        Load a language model and its tokenizer.

        Handles model quantization (bfloat16 on CUDA, float16 otherwise),
        device placement, and memory management.

        Args:
            model_name: Name or path of the model
            device: Device to load on (e.g., 'cuda', 'cpu')
            batch_size: Default batch size
            tokenize_kwargs: Additional tokenization args
            model_kwargs: Additional model init args
            tokenizer_kwargs: Additional tokenizer args

        Returns:
            Initialized LLMModel instance
        """
        hf_login()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": device,
            "torch_dtype": dtype,
            "cache_dir": CACHE_DIR,
            "max_memory": MODEL_MAX_MEMORY.get(model_name),
            **(model_kwargs or {}),
        }
        tokenizer_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "cache_dir": CACHE_DIR,
            **(tokenizer_kwargs or {}),
        }
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.generation_config.pad_token_id = tokenizer.pad_token_id

        default_tokenize_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": True,
            "max_length": 2**13,
        }

        tokenize_kwargs = default_tokenize_kwargs | (tokenize_kwargs or {})

        return cls(
            name=model_name,
            batch_size=batch_size,
            device=device,
            tokenize_kwargs=tokenize_kwargs,
            model=model,
            tokenizer=tokenizer,
        )

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

    def to(self, device: str) -> Self:
        self.device = device
        self.model.to(device)
        return self

    def tokenize(
        self, dialogues: Sequence[Input], add_generation_prompt: bool = False
    ) -> dict[str, torch.Tensor]:
        dialogues = [to_dialogue(d) for d in dialogues]
        input_dicts = [[d.model_dump() for d in dialogue] for dialogue in dialogues]

        input_str = self.tokenizer.apply_chat_template(
            input_dicts,
            tokenize=False,  # Return string instead of tokens
            add_generation_prompt=add_generation_prompt,  # Add final assistant prefix for generation
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
    def get_batched_activations_for_layers(
        self,
        dataset: BaseDataset,
        layers: list[int],
        batch_size: int = -1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Extract activations from multiple layers for a dataset.

        Processes inputs in batches, storing activations on CPU in float16
        to manage memory efficiently.

        Args:
            dataset: Input dataset
            layers: Layer indices to extract from
            batch_size: Processing batch size (-1 uses default)

        Returns:
            Tuple of:
            - Activations tensor [n_layers, n_samples, seq_len, hidden_dim]
            - Tokenized inputs dict
        """
        if batch_size == -1:
            batch_size = self.batch_size

        hidden_dim = self.model.config.hidden_size  # type: ignore
        assert isinstance(hidden_dim, int)
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
                batched_range(n_samples, batch_size),
                desc="Generating activations...",
            ):
                # Get batch of tokenized inputs
                batch_inputs = {k: v[start_idx:end_idx] for k, v in inputs.items()}

                # Get activations for this batch
                activations = hooked_model.get_acts(batch_inputs)

                # Write to the relevant slice of the big tensor
                all_activations[:, start_idx:end_idx] = activations

        return all_activations, inputs

    @torch.no_grad()
    def get_batched_activations(
        self,
        dataset: BaseDataset,
        layer: int,
        batch_size: int = -1,
    ) -> Activation:
        all_activations, inputs = self.get_batched_activations_for_layers(
            dataset, [layer], batch_size
        )
        return Activation(
            _activations=all_activations[0].numpy(),
            _attention_mask=inputs["attention_mask"].cpu().numpy(),
            _input_ids=inputs["input_ids"].cpu().numpy(),
        )

    @torch.no_grad()
    def compute_log_likelihood(
        self,
        inputs: Sequence[Input],
        batch_size: int = BATCH_SIZE,
    ) -> torch.Tensor:
        """
        Compute log probabilities for input sequences.

        Useful for evaluating model confidence and computing metrics
        like perplexity. Handles padding tokens properly.

        Args:
            inputs: Input sequences
            batch_size: Processing batch size

        Returns:
            Log probabilities tensor [n_samples, seq_len-1]
        """
        torch_inputs = self.tokenize(inputs)

        n_samples, seq_len = torch_inputs["input_ids"].shape

        # Create empty tensor for all log probabilities
        # We use seq_len-1 because we'll be shifting the targets by 1
        all_log_probs = torch.zeros(
            (n_samples, seq_len - 1), device="cpu", dtype=torch.float32
        )

        # Process in batches
        for start_idx, end_idx in tqdm(
            batched_range(n_samples, batch_size), desc="Computing log likelihoods..."
        ):
            # Get batch of tokenized inputs
            batch_inputs = {k: v[start_idx:end_idx] for k, v in torch_inputs.items()}

            # Forward pass through the model
            outputs = self.model(**batch_inputs)

            # Get logits and shift them to align predictions with targets
            logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
            targets = batch_inputs["input_ids"][:, 1:]  # (batch, seq_len-1)
            attention_mask = batch_inputs["attention_mask"][:, 1:]  # (batch, seq_len-1)

            # Compute log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Gather the log probs of the target tokens
            token_log_probs = log_probs.gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)

            # Mask out padding tokens
            token_log_probs = token_log_probs * attention_mask

            # Store batch results in the pre-allocated tensor
            all_log_probs[start_idx:end_idx] = token_log_probs.cpu()

        return all_log_probs

    def generate(
        self,
        dialogue: Dialogue,
        max_new_tokens: int = 10,
        temperature: float | None = None,
        do_sample: bool = False,
        top_p: float = 1.0,
        skip_special_tokens: bool = False,
        return_full_output: bool = False,
        **generation_kwargs: Any,
    ) -> str:
        """
        Generate text continuation for a dialogue.

        Handles tokenization, generation prompts, and decoding.
        Supports temperature, top-p sampling, and custom parameters.

        Args:
            dialogue: Input dialogue
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (None for greedy)
            do_sample: Use sampling instead of greedy
            top_p: Top-p sampling parameter
            skip_special_tokens: Skip special tokens in output
            return_full_output: Return full dialogue or just continuation
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        inputs = self.tokenize([dialogue], add_generation_prompt=True)

        # Generate the answer
        outputs = self.model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            **generation_kwargs,
        )

        if return_full_output:
            out_tokens = outputs[0]
        else:
            # Only get the newly generated tokens by slicing from the input length
            out_tokens = outputs[0][inputs["input_ids"].shape[1] :]

        return self.tokenizer.decode(
            out_tokens, skip_special_tokens=skip_special_tokens
        )
