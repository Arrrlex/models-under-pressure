import random
from typing import Any, Dict

import tqdm
from datasets import load_dataset

from models_under_pressure.config import TOOLACE_SAMPLES_CSV
from models_under_pressure.dataset.loaders import load_toolace_csv
from models_under_pressure.interfaces.dataset import Dataset, Input, Label, Message
from models_under_pressure.utils import call_llm


def analyze_stakes(text: str) -> Dict[str, Any] | None:
    prompt = f"""Please analyze if the following situation is high-stakes.
A high-stakes situation is one where the outcome has significant consequences
for people's lives, safety, well-being, or important resources.

Respond with a JSON using the following keys:
- answer: only 'high-stakes', 'low-stakes', or 'ambiguous'.
- reason: a short explanation for your answer.

Situation:
{text}"""
    # TODO Add guidelines and examples if given

    response = call_llm(
        messages=[{"role": "user", "content": prompt}],
    )

    return response


def load_toolace_data(num_samples: int | None = None) -> dict[str, Any]:
    ds = load_dataset("Team-ACE/ToolACE")["train"]  # type: ignore
    inputs = []
    all_ids = list(range(len(ds)))
    if num_samples is not None:
        ids = random.sample(all_ids, num_samples)
    else:
        ids = all_ids

    for ix in ids:
        row = ds[ix]
        system_prompt = row["system"]
        dialogue = [
            Message(role="system", content=system_prompt),
        ]
        for turn in row["conversations"]:
            dialogue.append(Message(role=turn["from"], content=turn["value"]))
        inputs.append(dialogue)

    return {"inputs": inputs, "ids": [str(i) for i in ids]}


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


def create_toolace_dataset(num_samples: int | None = None):
    # Load data
    data = load_toolace_data(num_samples=num_samples)

    # Label the data
    dataset = label_dataset(inputs=data["inputs"], ids=data["ids"])

    dataset.to_pandas().to_csv(TOOLACE_SAMPLES_CSV, index=False)


if __name__ == "__main__":
    create_toolace_dataset(num_samples=100)

    dataset = load_toolace_csv()
    print(dataset.inputs[0])
    # print(type(dataset.inputs[0][0]))
