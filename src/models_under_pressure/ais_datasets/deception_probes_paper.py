import json
from typing import Dict

from tqdm import tqdm

from models_under_pressure.config import DATA_DIR, PROJECT_ROOT
from models_under_pressure.eval_datasets.label_dataset import (
    LabelledDataset,
    label_dataset,
)
from models_under_pressure.interfaces.dataset import Dataset, Message


def gen_llama_rollouts() -> Dict[str, Dataset]:
    """
    Loads all JSON files containing 'llama-70b-3.3' in their names from the rollouts directory.
    Returns a dictionary mapping from the experiment name (e.g. 'insider_trading__onpolicy') to a Dataset object.
    Each Dataset contains:
    - inputs: List of conversations (each conversation is a list of Message objects)
    - ids: List of unique identifiers for each conversation
    - other_fields: Dictionary containing 'is_deceptive' field from graded_deception
    """
    rollouts_dir = PROJECT_ROOT.parent / "deception-detection" / "data" / "rollouts"
    result = {}

    # Dictionary to store AI audit datasets before combining
    ai_audit_datasets = []

    for file_path in rollouts_dir.glob("*llama-70b-3.3.json"):
        print(file_path)
        name = file_path.stem.replace("__llama-70b-3.3", "")

        data = json.loads(file_path.read_text())

        # Process each rollout into a record
        inputs = []
        ids = []
        deception_scores = []
        if not any(
            name.startswith(prefix)
            for prefix in ["ai_audit", "roleplaying", "ai_liar", "sandbagging"]
        ):
            continue

        pbar = tqdm(
            total=sum(
                1 for rollout in data["rollouts"] for _ in rollout["graded_deception"]
            )
        )

        for rollout_idx, rollout in enumerate(data["rollouts"]):
            input_messages = [
                Message(role=m["role"], content=m["content"])
                for m in rollout["input_messages"]
            ]

            if "sandbagging" in name:
                input_messages = input_messages[:-1]

            deception_results = rollout["graded_deception"]
            if any(isinstance(s, (int, float)) and s >= 2 for s in deception_results):
                deception_scores += [s / 7 for s in deception_results]
            elif isinstance(deception_results[0], str):
                deception_scores += [int(s == "deceptive") for s in deception_results]
            else:
                deception_scores += deception_results

            # Create a separate row for each output message
            for output_idx, output_msg in enumerate(rollout["output_messages"]):
                inputs.append(input_messages + [Message(**output_msg)])
                ids.append(f"{name}_{rollout_idx}_{output_idx}")
                pbar.update(1)
        pbar.close()
        # Create the Dataset
        dataset = Dataset(
            inputs=inputs, ids=ids, other_fields={"deception_scores": deception_scores}
        )

        # Label the dataset
        labelled_dataset = label_dataset(dataset, max_concurrent=500)

        # Store AI audit datasets separately for combining later
        if name.startswith("ai_audit"):
            ai_audit_datasets.append(labelled_dataset)
        else:
            result[name] = labelled_dataset

    # Combine AI audit datasets if we have them
    if ai_audit_datasets:
        result["ai_audit"] = LabelledDataset.concatenate(ai_audit_datasets)

    return result


def save_rollouts(rollouts: Dict[str, Dataset]) -> None:
    """
    Saves each rollout dataset to data/evals/ais/{name}.jsonl
    """
    ais_dir = DATA_DIR / "evals" / "ais"
    ais_dir.mkdir(parents=True, exist_ok=True)

    for name, dataset in rollouts.items():
        output_path = ais_dir / f"{name}.jsonl"
        print(f"Saving {name} to {output_path}")
        dataset.save_to(output_path, overwrite=True)


if __name__ == "__main__":
    rollouts = gen_llama_rollouts()
    save_rollouts(rollouts)
