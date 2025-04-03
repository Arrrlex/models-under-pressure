from typing import Any, Callable, Self, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models_under_pressure.config import BATCH_SIZE, CACHE_DIR, DEVICE, MODEL_MAX_MEMORY
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    Dialogue,
    Input,
    to_dialogue,
)
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

    @torch.no_grad()
    def compute_log_likelihood(
        self,
        inputs: Sequence[Input],
        batch_size: int = BATCH_SIZE,
    ) -> torch.Tensor:
        """
        Compute the log likelihoods for each input sequence with batching.

        Args:
            inputs: Sequence of Input objects
            batch_size: Size of batches to process at once

        Returns:
            torch.Tensor: Log likelihoods for each sequence, shape (n_samples, max_seq_len-1)
        """
        n_samples = len(inputs)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_log_probs = []
        max_seq_len = 0

        # Process in batches
        # for i in tqdm(range(n_batches), desc="Computing log likelihoods..."):
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_inputs = inputs[start_idx:end_idx]

            # Convert batch inputs to dialogues and tokenize
            batch_dialogues = [to_dialogue(inp) for inp in batch_inputs]
            torch_inputs = self.tokenize(batch_dialogues, add_generation_prompt=False)

            # Forward pass through the model
            outputs = self.model(
                input_ids=torch_inputs["input_ids"],
                attention_mask=torch_inputs["attention_mask"],
            )

            # Get logits and shift them to align predictions with targets
            logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
            targets = torch_inputs["input_ids"][:, 1:]  # (batch, seq_len-1)
            attention_mask = torch_inputs["attention_mask"][:, 1:]  # (batch, seq_len-1)

            # Compute log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Gather the log probs of the target tokens
            token_log_probs = log_probs.gather(
                dim=-1,
                index=targets.unsqueeze(-1),
            ).squeeze(-1)  # (batch, seq_len-1)

            # Mask out padding tokens
            token_log_probs = token_log_probs * attention_mask

            # Track maximum sequence length
            max_seq_len = max(max_seq_len, token_log_probs.shape[1])

            # Store batch results
            all_log_probs.append(token_log_probs)

        # Pad all batches to the same sequence length
        padded_log_probs = []
        for log_probs in all_log_probs:
            if log_probs.shape[1] < max_seq_len:
                padding = torch.zeros(
                    (log_probs.shape[0], max_seq_len - log_probs.shape[1]),
                    device=log_probs.device,
                )
                log_probs = torch.cat([log_probs, padding], dim=1)
            padded_log_probs.append(log_probs)

        # Combine all batches
        final_log_probs = torch.cat(padded_log_probs, dim=0)

        assert (
            final_log_probs.shape == (n_samples, max_seq_len)
        ), f"Expected log probs shape ({n_samples}, {max_seq_len}), got {final_log_probs.shape}"

        return final_log_probs

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
        input_str = self.tokenizer.apply_chat_template(
            [d.model_dump() for d in dialogue], tokenize=False
        )  # type: ignore

        tokenized = self.tokenizer(input_str, return_tensors="pt").to(self.model.device)  # type: ignore

        # Generate the answer
        outputs = self.model.generate(  # type: ignore
            **tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            **generation_kwargs,
        )

        if return_full_output:
            answer = self.tokenizer.decode(
                outputs[0], skip_special_tokens=skip_special_tokens
            )  # type: ignore
        else:
            # Only get the newly generated tokens by slicing from the input length
            new_tokens = outputs[0][tokenized.input_ids.shape[1] :]

            # Decode just the new tokens
            answer = self.tokenizer.decode(
                new_tokens, skip_special_tokens=skip_special_tokens
            )  # type: ignore

        return answer
