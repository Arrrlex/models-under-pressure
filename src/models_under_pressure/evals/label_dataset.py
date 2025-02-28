from typing import Any, Dict

import tqdm

from models_under_pressure.interfaces.dataset import Dataset, Input, Label
from models_under_pressure.utils import call_llm

# TODO This should be merged with dataset/metadata_generation


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
