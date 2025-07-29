import asyncio
import functools
import json
import os
import random
import string
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import indent
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    TypeVar,
    get_type_hints,
)

import httpx
import huggingface_hub
import hydra
import numpy as np
import openai
import requests
import torch
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import PreTrainedTokenizer

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")


def as_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert a torch.Tensor to a numpy array.
    """
    if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]:
        return x.detach().cpu().int().numpy()
    else:
        return x.detach().cpu().float().numpy()


class AttrDict(dict):
    def __init__(self, dict_: dict[str, Any]):
        super().__init__(**dict_)

    def __getattr__(self, key: str) -> Any:
        val = self[key]
        if isinstance(val, dict):
            return AttrDict(val)
        else:
            return val


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


T = TypeVar("T")


class pydra:
    @staticmethod
    def main(*args: Any, **kwargs: Any) -> Callable:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @hydra.main(*args, **kwargs)
            @functools.wraps(func)
            def wrapper(config: DictConfig) -> T:
                config_type = get_type_hints(func)["config"]
                config_dict = OmegaConf.to_container(
                    config, resolve=True, enum_to_str=True
                )
                config_model = config_type.model_validate(config_dict)
                return func(config_model)

            return wrapper

        return decorator


def _get_async_client() -> AsyncOpenAI:
    if not hasattr(_get_async_client, "_instance"):
        _get_async_client._instance = AsyncOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
    return _get_async_client._instance


def _get_openrouter_async_client() -> AsyncOpenAI:
    """Get async OpenRouter client with proper configuration."""
    raise NotImplementedError(
        "OpenRouter async client is now handled via httpx, not AsyncOpenAI."
    )


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


async def call_openrouter_async(
    messages: List[Any],
    model: str,
    json_schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    quantization: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async version of call_llm using OpenRouter API (httpx)"""
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY environment variable not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Arrrlex/models-under-pressure",
        "X-Title": "Models Under Pressure",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_schema is not None:
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if quantization is not None:
        # data["provider"] = {"quantization": quantization}
        data["provider"] = {"quantizations": [quantization]}
    data.update(kwargs)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
        )
        if resp.status_code != 200:
            raise ValueError(f"OpenRouter API error: {resp.status_code} - {resp.text}")
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError("No content returned from OpenRouter LLM")
        return json.loads(content)


async def call_openrouter_batch_async(
    messages_list: List[List[Any]],
    model: str,
    temperature: float | None = None,
    max_tokens: Optional[int] = None,
    max_concurrent: int = 10,
    task_timeout: Optional[float] = 60.0,
    quantization: Optional[str] = None,
    **kwargs: Any,
) -> List[str]:
    """
    Process multiple OpenRouter API requests concurrently (httpx).
    """
    import asyncio

    async def single_request(messages: List[Any]) -> str:
        return await call_openrouter_single_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            quantization=quantization,
            **kwargs,
        )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request(messages: List[Any]) -> str:
        async with semaphore:
            return await single_request(messages)

    bounded_tasks = [bounded_request(messages) for messages in messages_list]
    if task_timeout:
        results = await asyncio.gather(
            *[asyncio.wait_for(task, timeout=task_timeout) for task in bounded_tasks],
            return_exceptions=True,
        )
    else:
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i} failed: {result}")
            processed_results.append("")
        else:
            processed_results.append(result)
    return processed_results


async def call_openrouter_single_async(
    messages: List[Any],
    model: str,
    temperature: float | None = None,
    max_tokens: Optional[int] = None,
    quantization: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Make a single async OpenRouter API request (httpx).
    """
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY environment variable not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Arrrlex/models-under-pressure",
        "X-Title": "Models Under Pressure",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if quantization is not None:
        data["provider"] = {"quantizations": [quantization]}
    data.update(kwargs)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
        )
        if resp.status_code != 200:
            raise ValueError(f"OpenRouter API error: {resp.status_code} - {resp.text}")
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError("No content returned from OpenRouter LLM")
        return content


def call_openrouter_sync(
    messages: List[Any],
    model: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    quantization: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Synchronous version of OpenRouter API call for text generation (not JSON)"""
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Arrrlex/models-under-pressure",
        "X-Title": "Models Under Pressure",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    # Add quantization details if specified
    if quantization is not None:
        data["provider"] = {"quantizations": [quantization]}

    # Add any additional kwargs
    data.update(kwargs)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60,
    )

    if response.status_code != 200:
        raise ValueError(
            f"OpenRouter API error: {response.status_code} - {response.text}"
        )

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    if content is None:
        raise ValueError("No content returned from OpenRouter LLM")

    return content


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


def generate_short_id_with_timestamp(length: int = 4) -> str:
    """Generate a short, random ID using base62 encoding."""
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    with unset_random_seeds():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = "".join(random.choices(characters, k=length))
        return timestamp + "_" + random_id


def convert_paths(config: Any) -> Any:
    if isinstance(config, Path):
        return (
            config.relative_to(Path.cwd())
            if config.is_relative_to(Path.cwd())
            else config
        )
    elif isinstance(config, dict):
        return {k: convert_paths(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [convert_paths(item) for item in config]
    elif isinstance(config, BaseModel):
        for field in config.model_fields_set:
            setattr(config, field, convert_paths(getattr(config, field)))
        return config
    elif isinstance(config, DictConfig):
        return {k: convert_paths(v) for k, v in config.items()}
    return config


def pretty_format_config(config: Any) -> str:
    config = convert_paths(config)
    if isinstance(config, list) and isinstance(config[0], BaseModel):
        config = [config.model_dump(mode="json", by_alias=True) for config in config]
    if isinstance(config, BaseModel):
        config = config.model_dump(mode="json", by_alias=True)
    if isinstance(config, DictConfig):
        config = config.to_container()

    return yaml.dump(config, indent=2)


def double_check_config(config: Any, double_check: bool = True) -> None:
    print("Config:")
    print(indent(pretty_format_config(config), "  "))
    if double_check:
        is_ok = input("Do you really want to run this config? (y/n) ")
        if is_ok != "y" and is_ok != "Y":
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
    HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))
    if not HF_TOKEN:
        raise ValueError("No HuggingFace token found")
    try:
        huggingface_hub.login(token=HF_TOKEN)
    except requests.exceptions.HTTPError as e:
        print(
            f"Error logging in to HuggingFace: {e} (Might be fine in case of rate limit error.)"
        )
