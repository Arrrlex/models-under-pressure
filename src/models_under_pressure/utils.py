import json
import os
from typing import Any, Dict, List

import openai
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from models_under_pressure.config import DEFAULT_MODEL

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")


def call_llm(messages: List[Any], model: str | None = None) -> Dict[str, Any] | None:
    response = openai.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        return None
    return json.loads(content) if len(content) > 0 else None


def generate_completions(
    model: torch.nn.Module,
    tokenizer: "PreTrainedTokenizer",
    prompts: list[str],
    max_length: int = 1024,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    batch_size: int = -1,
):
    """
    Generate completions for a list of prompts using a given model and tokenizer.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use for generation.
        prompts: The prompts to generate completions for.
        max_length: The maximum length of the input prompts.
        max_new_tokens: The maximum number of tokens to generate.
        temperature: The temperature to use for generation.
        top_p: The top-p value to use for generation.
        top_k: The top-k value to use for generation.
        num_return_sequences: The number of return sequences to generate.
        do_sample: Whether to sample from the model.
        repetition_penalty: The repetition penalty to use for generation.
        batch_size: Batch size for processing prompts. If -1, process all prompts at once.

    Returns:
        A list of completions with prompts removed.
    """

    model.eval()
    all_completions = []

    # If batch_size is -1 or greater than number of prompts, process all at once
    effective_batch_size = len(prompts) if batch_size == -1 else batch_size

    with torch.no_grad():
        for i in tqdm(
            range(0, len(prompts), effective_batch_size), desc="Generating completions"
        ):
            batch_prompts = prompts[i : i + effective_batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )
            batch_completions = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            all_completions.extend(batch_completions)

    return all_completions
