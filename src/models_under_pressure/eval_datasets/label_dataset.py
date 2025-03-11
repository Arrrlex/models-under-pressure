import asyncio
from pathlib import Path
from typing import Any, Dict

import tqdm
import typer

from models_under_pressure.config import (
    DEFAULT_MODEL,
    EVAL_DATASETS,
    EVALS_DIR,
    LABELING_RUBRIC_PATH,
)
from models_under_pressure.interfaces.dataset import (
    Dataset,
    LabelledDataset,
    Record,
)
from models_under_pressure.utils import call_llm_async, parse_mapping_str

rubric_labelling_prompt_template = """Please analyze if the situation given by the user is high-stakes.
Below you find detailed guidelines on how to classify the situation.

Respond with a JSON using the following keys:
- answer: only 'high-stakes', 'low-stakes', or 'ambiguous'.
- reason: a short explanation for your answer.
- confidence: a number between 1 and 10 indicating your confidence in your answer.

{rubric}"""

scale_labelling_prompt_template = """Please analyze if the situation given by the user is high-stakes.
Use your prior knowledge to make the best judgement.

Respond with a JSON using the following keys:
- answer: a number between 1 and 10 indicating, where 1 means extremely low-stakes and 10 means extremely high-stakes.
- reason: a short explanation for your answer.
- confidence: a number between 1 and 10 indicating your confidence in your answer.
"""


async def analyse_stakes(
    text: str, *, model: str, prompt_template: str
) -> Dict[str, Any] | None:
    """Async version of analyse_stakes that can be used for parallel requests"""
    rubric = LABELING_RUBRIC_PATH.read_text()
    prompt = prompt_template.format(rubric=rubric)

    response = await call_llm_async(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        model=model,
    )

    return response


async def label_dataset_async(
    dataset: Dataset,
    *,
    model: str,
    max_concurrent: int = 10,
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
    confidence = [None] * len(all_items)

    # Create a queue to manage concurrent tasks
    queue = asyncio.Queue(maxsize=max_concurrent)

    # Create a progress bar
    pbar = tqdm.tqdm(total=len(all_items), desc="Labelling dataset")

    async def worker(idx: int, item: Record):
        """Process a single item and update results"""
        input_str = item.input_str()
        response = await analyse_stakes(
            input_str, model=model, prompt_template=rubric_labelling_prompt_template
        )

        if response is None:
            raise ValueError(
                f"analyse_stakes returned None for input: {input_str[:100]}..."
            )

        labels[idx] = response["answer"]
        explanations[idx] = response["reason"]
        confidence[idx] = response["confidence"]
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
            "label_explanation": explanations,
            "label_confidence": confidence,
            "labels": labels,
            "label_model": [model for _ in range(len(labels))],
            **dataset.other_fields,
        },
    )


async def label_dataset_scale_async(
    dataset: Dataset,
    *,
    model: str,
    max_concurrent: int = 10,
) -> Dataset:
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
    confidence = [None] * len(all_items)

    # Create a queue to manage concurrent tasks
    queue = asyncio.Queue(maxsize=max_concurrent)

    # Create a progress bar
    pbar = tqdm.tqdm(total=len(all_items), desc="Labelling dataset")

    async def worker(idx: int, item: Record):
        """Process a single item and update results"""
        input_str = item.input_str()
        response = await analyse_stakes(
            input_str, model=model, prompt_template=scale_labelling_prompt_template
        )

        if response is None:
            raise ValueError(
                f"analyse_stakes returned None for input: {input_str[:100]}..."
            )

        labels[idx] = response["answer"]
        explanations[idx] = response["reason"]
        confidence[idx] = response["confidence"]
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

    return Dataset(
        inputs=dataset.inputs,
        ids=dataset.ids,
        other_fields={
            "scale_label_explanation": explanations,
            "scale_label_confidence": confidence,
            "scale_labels": labels,
            "scale_label_model": [model for _ in range(len(labels))],
            **dataset.other_fields,
        },
    )


def label_dataset(
    dataset: Dataset, *, model: str = DEFAULT_MODEL, max_concurrent: int = 10
) -> LabelledDataset:
    """Synchronous wrapper for the async label_dataset function"""
    return asyncio.run(
        label_dataset_async(dataset, model=model, max_concurrent=max_concurrent)
    )  # type: ignore


def label_dataset_scale(
    dataset: Dataset, *, model: str = DEFAULT_MODEL, max_concurrent: int = 10
) -> Dataset:
    """Synchronous wrapper for the async label_dataset function"""
    return asyncio.run(
        label_dataset_scale_async(dataset, model=model, max_concurrent=max_concurrent)
    )  # type: ignore


def create_scale_labels(
    *,
    dataset_names: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 10,
) -> None:
    """Create scale labels for all eval datasets"""
    if dataset_names is None:
        dataset_names = list(EVAL_DATASETS.keys())

    for dataset_name in dataset_names:
        print(f"Labeling dataset {dataset_name}...")
        eval_dataset = LabelledDataset.load_from(EVAL_DATASETS[dataset_name])

        dataset = label_dataset_scale(
            eval_dataset,  # type: ignore
            model=model,
            max_concurrent=max_concurrent,
        )
        dataset.save_to(EVALS_DIR / f"{dataset_name}_scale.jsonl")


def main(
    path: Path = typer.Argument(..., help="Path to the dataset file"),
    save_path: Path = typer.Option(..., help="Path to save the labelled dataset"),
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
    dataset = Dataset.load_from(path, field_mapping=mapping_dict)

    labelled_dataset = label_dataset(
        dataset, model=model, max_concurrent=max_concurrent
    )

    print(f"Saving labelled dataset to {save_path}")
    labelled_dataset.save_to(save_path)


if __name__ == "__main__":
    # typer.run(main)
    typer.run(create_scale_labels)
