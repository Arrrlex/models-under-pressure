import asyncio
from pathlib import Path
from typing import Any, Dict

import tqdm
import typer

from models_under_pressure.config import DEFAULT_MODEL, LABELING_RUBRIC_PATH
from models_under_pressure.interfaces.dataset import (
    Dataset,
    LabelledDataset,
    Record,
)
from models_under_pressure.utils import call_llm_async, parse_mapping_str

labelling_prompt_template = """Please analyze if the following situation is high-stakes.
Below you find detailed guidelines on how to classify the situation.

Respond with a JSON using the following keys:
- answer: only 'high-stakes', 'low-stakes', or 'ambiguous'.
- reason: a short explanation for your answer.

{rubric}"""


async def analyse_stakes(text: str, *, model: str) -> Dict[str, Any] | None:
    """Async version of analyse_stakes that can be used for parallel requests"""
    rubric = LABELING_RUBRIC_PATH.read_text()
    prompt = labelling_prompt_template.format(text=text, rubric=rubric)

    response = await call_llm_async(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Situation:\n{text}"},
        ],
        model=model,
    )

    return response


async def label_dataset_async(
    dataset: Dataset, *, model: str, max_concurrent: int = 10
) -> LabelledDataset:
    """
    Asynchronously label a dataset using a queue to limit concurrency.

    Args:
        dataset: The dataset to label
        model: The model to use for labeling
        max_concurrent: Maximum number of concurrent API calls
    """
    all_items = dataset.to_records()
    labels = [None] * len(all_items)
    explanations = [None] * len(all_items)

    # Create a queue to manage concurrent tasks
    queue = asyncio.Queue(maxsize=max_concurrent)

    # Create a progress bar
    pbar = tqdm.tqdm(total=len(all_items), desc="Labelling dataset")

    async def worker(idx: int, item: Record):
        """Process a single item and update results"""
        input_str = item.input_str()
        response = await analyse_stakes(input_str, model=model)

        if response is None:
            raise ValueError(
                f"analyse_stakes returned None for input: {input_str[:100]}..."
            )

        labels[idx] = response["answer"]
        explanations[idx] = response["reason"]
        pbar.update(1)
        await queue.get()  # Signal task completion

    # Start processing all items
    tasks = []
    for idx, item in enumerate(all_items):
        await queue.put(idx)  # Wait if queue is full
        task = asyncio.create_task(worker(idx, item))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    pbar.close()

    return LabelledDataset(
        inputs=dataset.inputs,
        ids=dataset.ids,
        other_fields={
            "explanation": explanations,
            "labels": labels,
            **dataset.other_fields,
        },
    )


def label_dataset(
    dataset: Dataset, *, model: str = DEFAULT_MODEL, max_concurrent: int = 10
) -> LabelledDataset:
    """Synchronous wrapper for the async label_dataset function"""
    return asyncio.run(
        label_dataset_async(dataset, model=model, max_concurrent=max_concurrent)
    )


def main(
    path: Path = typer.Argument(..., help="Path to the dataset file"),
    save_path: Path = typer.Option(..., help="Path to save the labelled dataset"),
    split: str = typer.Option("train", help="Dataset split to use"),
    field_mapping: str = typer.Option(
        "",
        help="Comma-separated list of key:value pairs for field mapping (e.g., 'input:text,id:example_id')",
    ),
    model: str = typer.Option(DEFAULT_MODEL, help="Model to use for labelling"),
    max_concurrent: int = typer.Option(
        10, help="Maximum number of concurrent API calls"
    ),
):
    """
    Label a dataset by analysing whether each situation is high-stakes or low-stakes.

    The field_mapping parameter allows mapping dataset fields to expected fields.
    Example: 'input:text,id:example_id' maps the 'text' field to 'input' and 'example_id' to 'id'.
    """
    # Parse field_mapping string into a dictionary
    mapping_dict = parse_mapping_str(field_mapping)

    print(f"Loading dataset from {path}")
    dataset = Dataset.load_from(path, split=split, field_mapping=mapping_dict)

    labelled_dataset = label_dataset(
        dataset, model=model, max_concurrent=max_concurrent
    )

    print(f"Saving labelled dataset to {save_path}")
    labelled_dataset.save_to(save_path)


if __name__ == "__main__":
    typer.run(main)
