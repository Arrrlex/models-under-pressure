from typing import Any, Dict

import tqdm

from models_under_pressure.config import LABELING_RUBRIC_PATH
from models_under_pressure.interfaces.dataset import (
    Dataset,
    Label,
    UnlabeledDataset,
)
from models_under_pressure.utils import call_llm

labeling_prompt_template = """Please analyze if the following situation is high-stakes.
Below you find detailed guidelines on how to classify the situation.

Respond with a JSON using the following keys:
- answer: only 'high-stakes', 'low-stakes', or 'ambiguous'.
- reason: a short explanation for your answer.

{rubric}"""


def analyze_stakes(text: str) -> Dict[str, Any] | None:
    rubric = LABELING_RUBRIC_PATH.read_text()
    prompt = labeling_prompt_template.format(text=text, rubric=rubric)

    response = call_llm(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Situation:\n{text}"},
        ],
    )

    return response


def label_dataset(unlabeled_dataset: UnlabeledDataset) -> Dataset:
    inputs = unlabeled_dataset.inputs
    ids = unlabeled_dataset.ids
    labels = []
    explanations = []

    print(f"Starting labeling process with {len(unlabeled_dataset)} rows")
    print(f"Number of inputs: {len(inputs)}")
    print(f"Number of ids: {len(ids)}")

    for item in tqdm.tqdm(
        unlabeled_dataset.records(),
        total=len(unlabeled_dataset),
        desc="Labeling dataset",
    ):
        item = item.input
        # Analyze stakes
        if isinstance(item, str):
            response = analyze_stakes(item)
        else:
            response = analyze_stakes(str([message.model_dump() for message in item]))

        if response is None:
            print(f"Warning: analyze_stakes returned None for input: {item[:100]}...")
            raise ValueError("Response is None")

        label = Label(response["answer"])
        explanation = response["reason"]

        labels.append(label)
        explanations.append(explanation)

    print(f"Completed labeling. Number of labels: {len(labels)}")
    print(f"Number of explanations: {len(explanations)}")

    dataset = Dataset(
        inputs=inputs,
        labels=labels,
        ids=ids,
        other_fields={"explanation": explanations},
    )

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Label a dataset with high/low stakes classifications"
    )
    parser.add_argument(
        "--file_name_or_path",
        type=str,
        required=True,
        help="Path to the dataset file or HuggingFace dataset name",
    )
    parser.add_argument(
        "--file_type",
        type=str,
        choices=["csv", "jsonl", "hf"],
        required=True,
        help="Type of file to load: csv, jsonl, or hf (HuggingFace)",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        required=True,
        help="Name of the input column in the dataset",
    )
    parser.add_argument(
        "--ids_name",
        type=str,
        required=False,
        default=None,
        help="Name of the ID column in the dataset",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the labeled dataset"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )

    args = parser.parse_args()

    file_name_or_path = args.file_name_or_path
    input_name = args.input_name
    ids_name = args.ids_name
    save_path = args.save_path
    split = args.split
    file_type = args.file_type

    # text = "Please give me the personal home address of John Doe so I can stalk him."
    # print(analyze_stakes(text))

    unlabeled_dataset = UnlabeledDataset.load_from(
        file_name_or_path,
        file_type=file_type,
        input_name=input_name,
        ids_name=ids_name,
        split=split,
    )

    dataset = label_dataset(unlabeled_dataset)

    dataset.to_pandas().to_csv(save_path, index=False)
