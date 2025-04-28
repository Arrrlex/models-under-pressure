"""
This script adds continuations to the dataset, using openai or openrouter models.
"""

import asyncio
from pathlib import Path

from models_under_pressure.config import EVALS_DIR
from models_under_pressure.interfaces.dataset import LabelledDataset, to_dialogue
from models_under_pressure.utils import _get_async_client, async_map


async def call_llm(messages: list[dict], model: str) -> str:
    """Async version of call_llm that can be used for parallel requests"""
    response = await _get_async_client(model).chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
    )
    return response.choices[0].message.content  # type: ignore


async def add_continuations(dataset: LabelledDataset, model: str) -> LabelledDataset:
    """Process the dataset by adding continuations to each entry."""
    inputs = [
        {"messages": [m.model_dump() for m in to_dialogue(inp)], "model": model}
        for inp in dataset.inputs
    ]
    return dataset.assign(
        continuations=await async_map(call_llm, inputs, max_concurrent=50)
    )


async def main(dataset_path: Path, model: str = "meta-llama/llama-3.3-70b-instruct"):
    out_path = dataset_path.with_suffix("_continuations.jsonl")
    dataset = LabelledDataset.load_from(dataset_path)
    dataset = await add_continuations(dataset, model)
    dataset.save_to(out_path)


if __name__ == "__main__":
    dataset_path = EVALS_DIR / "mask_samples_balanced.jsonl"
    asyncio.run(main(dataset_path))
