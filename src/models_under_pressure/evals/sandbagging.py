# %%

import json
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


# # function to create user prompt that will ask the model to answer a question taken from examples and giving 4 choices. the question is in examples[i]["question"] and the choices are in examples[i]["choices"]
def create_user_prompt(sample: dict) -> str:
    question = sample["question"]
    choices = sample["choices"]
    return f"Question: {question}\nChoices:\n{'\n'.join([f'({ix + 1}) {choice}' for ix, choice in enumerate(choices)])} \n Please reply only and only with the number of the correct answer!"


def generate_answers(
    examples: list[dict],
    system_prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    output_file: Path | None = None,
) -> list[dict]:
    answers = []

    print(len(examples))
    print(examples[0])

    for i in range(
        len(examples)
    ):  # Process first 3 examples or fewer if dataset is smaller
        # Format the input
        user_prompt = create_user_prompt(examples[i])
        tokenizer_ques = tokenizer.apply_chat_template(  # type: ignore
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
        )
        inputs = tokenizer(tokenizer_ques, return_tensors="pt")  # type: ignore

        # Generate the answer
        outputs = model.generate(  # type: ignore
            **inputs, max_new_tokens=8, temperature=0.0, do_sample=False
        )

        # Decode the answer
        # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # model_response = answer[len(tokenizer_ques):]
        model_response = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        # Print the question and answer
        print(f"Question: {user_prompt}")
        # Extract just the number from the response using regex

        match = re.search(r"\d+", model_response)
        choice = match.group() if match else None

        answer_field = "answer" if "answer" in examples[i] else "correct_answer"

        if choice is not None:
            choice = (
                int(choice) - 1
            )  # In the system prompt the answer is 1,2,3,4 but in the dataset it is 0,1,2,3

            res = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "question": examples[i]["question"],
                "choices": examples[i]["choices"],
                "model_response": model_response,
                "correct_answer": examples[i][answer_field],
                "model_answer": choice,
                "is_correct": choice == examples[i][answer_field],
                "subject": examples[i]["subject"],
            }
            if output_file is not None:
                with open(output_file, "a") as f:
                    f.write(json.dumps(res) + "\n")

            if choice == int(examples[i][answer_field]):
                print("HELL YEAH!")

            answers.append(res)
        print(f"Generated answer: {choice}")
        print("-" * 50)
    return answers
