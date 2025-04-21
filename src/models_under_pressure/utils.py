import asyncio
import functools
import json
import os
import random
import string
from textwrap import indent
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
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

import huggingface_hub
import hydra
import numpy as np
import openai
from pydantic import BaseModel
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI
from tqdm import tqdm
import yaml

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")


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
    HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))
    if not HF_TOKEN:
        raise ValueError("No HuggingFace token found")
    huggingface_hub.login(token=HF_TOKEN)
