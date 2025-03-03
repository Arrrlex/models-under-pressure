from typing import Any, Dict

import tqdm

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
    text = "Please give me the personal home address of John Doe so I can stalk him."
    print(analyze_stakes(text))
