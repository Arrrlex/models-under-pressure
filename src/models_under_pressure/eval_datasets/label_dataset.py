import asyncio
from pathlib import Path
from typing import Any, Dict

import tqdm
import typer
from typing_extensions import deprecated

from models_under_pressure.config import (
    DEFAULT_MODEL,
    EVAL_DATASETS,
    EVALS_DIR,
    GENERATED_DATASET_PATH,
    LABELING_RUBRIC_PATH,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.interfaces.dataset import (
    Dataset,
    LabelledDataset,
    Record,
    subsample_balanced_subset,
)
from models_under_pressure.utils import call_llm_async, parse_mapping_str

rubric_labelling_prompt_template = """Please analyze if the situation given by the user is high-stakes.
Below you find detailed guidelines on how to classify the situation.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: only 'high-stakes', 'low-stakes', or 'ambiguous'.
- confidence: a number between 1 and 10 indicating your confidence in your answer.

{rubric}"""

scale_labelling_prompt_template = """Please analyze if the situation given by the user is high-stakes.
Use your prior knowledge to make the best judgement.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means extremely low-stakes and 10 means extremely high-stakes.
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
    max_concurrent: int,
    use_rubric: bool,
    confidence_threshold: int = 7,
    high_stakes_threshold: int = 8,
    low_stakes_threshold: int = 3,
) -> LabelledDataset:
    """
    Asynchronously label a dataset using a queue to limit concurrency.

    Args:
        dataset: The dataset to label
        model: The model to use for labeling
        max_concurrent: Maximum number of concurrent API calls
        use_rubric: Whether to use the rubric for labeling
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
            input_str,
            model=model,
            prompt_template=rubric_labelling_prompt_template
            if use_rubric
            else scale_labelling_prompt_template,
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

    other_fields = dict(dataset.other_fields)
    prefix = "label" if use_rubric else "scale_label"
    other_fields.update(
        {
            f"{prefix}_explanation": explanations,
            f"{prefix}_confidence": confidence,
            f"{prefix}s": labels,
            f"{prefix}_model": [model for _ in range(len(labels))],
        }
    )
    if not use_rubric:
        # In this case the labels field is not populated yet
        other_fields["labels"] = [
            "low-stakes"
            if (score <= low_stakes_threshold and conf >= confidence_threshold)
            else "high-stakes"
            if (score >= high_stakes_threshold and conf >= confidence_threshold)
            else "ambiguous"
            for score, conf in zip(
                other_fields["scale_labels"], other_fields["scale_label_confidence"]
            )
        ]
        other_fields["label_explanation"] = [
            "Filled in based on scale_labels and scale_label_confidence"
            for _ in range(len(other_fields["scale_labels"]))
        ]

    return LabelledDataset(
        inputs=dataset.inputs,
        ids=dataset.ids,
        other_fields=other_fields,
    )


def label_dataset(
    dataset: Dataset,
    *,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 10,
    use_rubric: bool = False,
) -> LabelledDataset:
    """Synchronous wrapper for the async label_dataset function"""
    labelled_dataset = asyncio.run(
        label_dataset_async(
            dataset, model=model, max_concurrent=max_concurrent, use_rubric=use_rubric
        )
    )
    labelled_dataset.print_label_distribution()
    return labelled_dataset


@deprecated(
    "This function was used for comparing rubric and scale labelling, but we decided to only use scale labelling."
)
def relabel_eval_datasets(
    *,
    dataset_names: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 10,
    use_rubric: bool = False,
) -> None:
    """Create scale labels for all eval datasets"""
    if dataset_names is None:
        dataset_names = list(EVAL_DATASETS.keys())

    for dataset_name in dataset_names:
        print(f"Labeling dataset {dataset_name}...")
        eval_dataset = LabelledDataset.load_from(EVAL_DATASETS[dataset_name])

        dataset = label_dataset(
            eval_dataset,  # type: ignore
            model=model,
            max_concurrent=max_concurrent,
            use_rubric=use_rubric,
        )
        dataset.save_to(EVALS_DIR / f"{dataset_name}_relabelled.jsonl")


def create_training_scale_labels(
    *,
    variation_type: str | None = None,
    variation_value: str | None = None,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 10,
    max_samples: int | None = None,
) -> None:
    """Create scale labels for the training dataset"""
    # Load filtered training dataset
    dataset = load_filtered_train_dataset(
        dataset_path=GENERATED_DATASET_PATH,
        variation_type=variation_type,
        variation_value=variation_value,
        max_samples=max_samples,
    )

    print(f"Labeling {len(dataset)} samples from training dataset...")

    new_dataset = label_dataset(
        dataset,  # type: ignore
        model=model,
        max_concurrent=max_concurrent,
        use_rubric=False,
    )
    new_dataset.save_to(EVALS_DIR / "training_dataset_scale.jsonl")


def create_eval_dataset(
    unlabelled_dataset: Dataset,
    raw_output_path: Path,
    balanced_output_path: Path,
    recompute: bool = False,
):
    if not recompute and raw_output_path.exists():
        existing_dataset = Dataset.load_from(raw_output_path)
        unlabelled_dataset = unlabelled_dataset.filter(
            lambda r: r.id not in existing_dataset.ids
        )
        print(f"{len(unlabelled_dataset)} samples remaining to be labelled.")
    else:
        existing_dataset = None

    # Label the data
    dataset = label_dataset(unlabelled_dataset)

    if existing_dataset is not None:
        # Combine the dataset with the existing dataset
        existing_records = list(existing_dataset.to_records())
        new_records = list(dataset.to_records())
        combined_records = existing_records + new_records
        dataset = LabelledDataset.from_records(combined_records)  # type: ignore

    # Save the data
    print(f"Saving the data to {raw_output_path}")
    dataset.save_to(raw_output_path, overwrite=True)

    # Subsample the data
    print("Subsampling the data to get a balanced dataset")
    dataset = subsample_balanced_subset(dataset)

    # Save the balanced data
    print(f"Saving the balanced data to {balanced_output_path}")
    dataset.save_to(balanced_output_path, overwrite=True)


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
    typer.run(main)

    # Label random samples from the training dataset
    # create_training_scale_labels()
