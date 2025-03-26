import asyncio
from contextlib import contextmanager
import json
import os
import random
import string
import time
from datetime import timedelta
from pprint import pformat
from typing import Any, Awaitable, Callable, Dict, Generator, List, Optional, Sequence

import numpy as np

import huggingface_hub
import openai
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import PreTrainedTokenizer

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")


def _get_async_client() -> AsyncOpenAI:
    if not hasattr(_get_async_client, "_instance"):
        _get_async_client._instance = AsyncOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
    return _get_async_client._instance


# TODO Change messages type to Dialogue type?
def call_llm(messages: List[Any], model: str) -> Dict[str, Any] | None:
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        return None
    return json.loads(content) if len(content) > 0 else None


async def call_llm_async(
    messages: List[Any],
    model: str,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version of call_llm that can be used for parallel requests"""
    client = _get_async_client()
    if json_schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
    else:
        response_format = {"type": "json_object"}

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,  # type: ignore
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("No content returned from LLM")
    return json.loads(content)


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


def parse_mapping_str(mapping_str: str) -> dict[str, str]:
    mapping_dict = {}
    for pair in mapping_str.split(","):
        if ":" in pair:
            key, value = pair.split(":", 1)
            mapping_dict[key.strip()] = value.strip()
    return mapping_dict


async def async_map(
    func: Callable[..., Awaitable[Any]],
    items: Sequence[Any],
    max_concurrent: int,
    with_pbar: bool = True,
    task_timeout: Optional[float] = None,
) -> list[Any]:
    """
    Asynchronously map a function over a sequence of items with concurrency control.

    Args:
        func: Async function to apply to each item
        items: Sequence of items to process
        max_concurrent: Maximum number of concurrent tasks
        with_pbar: Whether to show a progress bar
        task_timeout: Optional timeout in seconds for each task

    Returns:
        List of results in the same order as input items
    """
    if with_pbar:
        pbar = tqdm(
            total=len(items),
            desc=f"Running {func.__name__} on {len(items)} items ...",
        )
    else:
        pbar = None

    sem = asyncio.Semaphore(max_concurrent)
    results = []

    async def worker(item: Any) -> Any:
        async with sem:
            try:
                if task_timeout is not None:
                    result = await asyncio.wait_for(func(**item), timeout=task_timeout)
                else:
                    result = await func(**item)
                if pbar is not None:
                    pbar.update(1)
                return result
            except (asyncio.TimeoutError, Exception) as e:
                print(f"Task failed with error: {e}")
                if pbar is not None:
                    pbar.update(1)
                return None

    # Create and gather all tasks
    tasks = [asyncio.create_task(worker(item)) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    if pbar is not None:
        pbar.close()

    return [r for r in results if r is not None]


@contextmanager
def unset_random_seeds():
    """
    Context manager that temporarily unsets random seeds from random, torch, and numpy,
    then restores them after the context block.
    """
    # Store current states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    # Unset seeds by using time-based seeds
    current_time = int(time.time() * 1000)
    # Ensure numpy seed is within valid range (0 to 2**32 - 1)
    numpy_seed = current_time % (2**32)

    random.seed(current_time)
    np.random.seed(numpy_seed)
    torch.manual_seed(current_time)

    try:
        yield
    finally:
        # Restore original states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def generate_short_id(length: int = 8) -> str:
    """Generate a short, random ID using base62 encoding."""
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    with unset_random_seeds():
        return "".join(random.choices(characters, k=length))


def double_check_config(config: Any) -> None:
    print(f"Config: {pformat(config)}")
    is_ok = input("Do you really want to run this config? (y/n) ")
    if is_ok != "y":
        raise ValueError("Config not confirmed")


def print_progress(
    iter_: Sequence[Any], report_every: int = 1
) -> Generator[Any, None, None]:
    """
    Wrapper around an iterator that prints progress and estimated time remaining.

    Like tqdm, but better for e.g. tmux where progress bars often look weird.
    """
    start_time = time.time()
    n = len(iter_)
    for i, item in enumerate(iter_):
        i += 1
        yield item
        if i % report_every == 0:
            elapsed = time.time() - start_time
            items_per_sec = i / elapsed
            remaining_items = n - i
            est_remaining = remaining_items / items_per_sec
            print(
                f"Progress: {i}/{n} | "
                f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                f"Remaining: {timedelta(seconds=int(est_remaining))}"
            )


def hf_login():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))
    if not HF_TOKEN:
        raise ValueError("No HuggingFace token found")
    huggingface_hub.login(token=HF_TOKEN)
