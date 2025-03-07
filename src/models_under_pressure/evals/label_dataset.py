from pathlib import Path
from typing import Any, Dict

import tqdm
import typer

from models_under_pressure.config import DEFAULT_MODEL, LABELING_RUBRIC_PATH
from models_under_pressure.interfaces.dataset import (
    Dataset,
    LabelledDataset,
)
from models_under_pressure.utils import call_llm

labelling_prompt_template = """Please analyze if the following situation is high-stakes.
Below you find detailed guidelines on how to classify the situation.

Respond with a JSON using the following keys:
- answer: only 'high-stakes', 'low-stakes', or 'ambiguous'.
- reason: a short explanation for your answer.

{rubric}"""


def analyse_stakes(text: str, *, model: str) -> Dict[str, Any] | None:
    rubric = LABELING_RUBRIC_PATH.read_text()
    prompt = labelling_prompt_template.format(text=text, rubric=rubric)

    response = call_llm(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Situation:\n{text}"},
        ],
        model=model,
    )

    return response


def label_dataset(dataset: Dataset, *, model: str) -> LabelledDataset:
    labels = []
    explanations = []

    for item in tqdm.tqdm(
        dataset.to_records(),
        total=len(dataset),
        desc="Labelling dataset",
    ):
        input_str = item.input_str()
        response = analyse_stakes(input_str, model=model)

        if response is None:
            raise ValueError(
                f"analyse_stakes returned None for input: {input_str[:100]}..."
            )

        labels.append(response["answer"])
        explanations.append(response["reason"])

    return LabelledDataset(
        inputs=dataset.inputs,
        ids=dataset.ids,
        other_fields={
            "explanation": explanations,
            "labels": labels,
            **dataset.other_fields,
        },
    )


def main(
    file_name_or_path: Path = typer.Argument(..., help="Path to the dataset file"),
    save_path: Path = typer.Option(..., help="Path to save the labelled dataset"),
    split: str = typer.Option("train", help="Dataset split to use"),
    field_mapping: str = typer.Option(
        "",
        help="Comma-separated list of key:value pairs for field mapping (e.g., 'input:text,id:example_id')",
    ),
    model: str = typer.Option(DEFAULT_MODEL, help="Model to use for labelling"),
):
    """
    Label a dataset by analysing whether each situation is high-stakes or low-stakes.

    The field_mapping parameter allows mapping dataset fields to expected fields.
    Example: 'input:text,id:example_id' maps the 'text' field to 'input' and 'example_id' to 'id'.
    """
    # Parse field_mapping string into a dictionary
    mapping_dict = {}
    if field_mapping:
        for pair in field_mapping.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                mapping_dict[key.strip()] = value.strip()

    dataset = Dataset.load_from(
        file_name_or_path,
        split=split,
        field_mapping=mapping_dict,
    )

    labelled_dataset = label_dataset(dataset, model=model)

    print(f"Saving labelled dataset to {save_path}")
    labelled_dataset.save_to(save_path)


if __name__ == "__main__":
    typer.run(main)
