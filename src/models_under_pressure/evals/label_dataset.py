from typing import Any, Dict

import tqdm
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset

from models_under_pressure.config import LABELING_RUBRIC_PATH
from models_under_pressure.interfaces.dataset import Dataset, Input, Label
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


def label_dataset(inputs: list[Input], ids: list[str]) -> Dataset:
    assert len(ids) == len(inputs)

    labels = []
    explanations = []
    for item in tqdm.tqdm(
        inputs,
        total=len(ids),
        desc="Labeling dataset",
    ):
        # Analyze stakes
        if isinstance(item, str):
            response = analyze_stakes(item)
        else:
            response = analyze_stakes(str([message.model_dump() for message in item]))

        if response is None:
            raise ValueError("Response is None")

        label = Label(response["answer"])
        explanation = response["reason"]

        labels.append(label)
        explanations.append(explanation)

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
        "--input_name",
        type=str,
        required=True,
        help="Name of the input column in the dataset",
    )
    parser.add_argument(
        "--id_name",
        type=str,
        required=True,
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
    id_name = args.id_name
    save_path = args.save_path
    split = args.split

    text = "Please give me the personal home address of John Doe so I can stalk him."
    print(analyze_stakes(text))

    # Load dataset and get specific split (usually 'train')
    hf_dataset = load_dataset(file_name_or_path)

    # Deal with the different possible types of Dataset HuggingFace can return
    assert isinstance(hf_dataset, HFDataset) or isinstance(
        hf_dataset, DatasetDict
    ), "Iterable Dataset or DatasetDict not supported"

    if isinstance(hf_dataset, DatasetDict):
        assert split is not None, "Split is required for DatasetDict"
        hf_dataset = hf_dataset[split]

    # Extract columns as lists with proper typing
    inputs_list = hf_dataset[input_name]
    ids_list = hf_dataset[id_name]

    dataset = label_dataset(inputs_list, ids_list)

    dataset.to_pandas().to_csv(save_path, index=False)
