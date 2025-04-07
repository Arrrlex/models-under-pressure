from models_under_pressure.config import DATA_DIR, EVAL_DATASETS
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import LabelledDataset, Message, Record

# TODO! Create another script which creates labelled datasets with different preprocessing and labelling according to my google doc

labelling_model = "gpt-4o-mini"
out_dir = DATA_DIR / "results" / "labelling_comparison"
max_samples = 10

anthropic_system_prompt = "You are an LLM as part of a general purpose chatbot."
mt_system_prompt = "You are an LLM as part of a medical assistant system, interacting with medical practitioners to improve efficiency."


def add_system_prompt(x: Record, system_prompt: str) -> Record:
    if isinstance(x.input, str):
        x.input = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=x.input),
        ]
    else:  # Dialogue type
        x.input = [Message(role="system", content=system_prompt)] + x.input  # type: ignore
    return x


settings = [
    {
        "name": "anthropic_system_context",
        "dataset": "anthropic",
        "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
        "labelling_method": "scale",
    },
    {
        "name": "anthropic_system_and_labelling_context",
        "dataset": "anthropic",
        "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
        "labelling_method": "anthropic_context",
    },
    {
        "name": "anthropic_system_and_complex_labelling_context",
        "dataset": "anthropic",
        "preprocessing_fn": lambda x: add_system_prompt(x, anthropic_system_prompt),
        "labelling_method": "anthropic_extended_context",
    },
    {
        "name": "mt_system_context",
        "dataset": "mt",
        "preprocessing_fn": lambda x: add_system_prompt(x, mt_system_prompt),
        "labelling_method": "scale",
    },
    {
        "name": "mt_system_and_labelling_context",
        "dataset": "mt",
        "preprocessing_fn": lambda x: add_system_prompt(x, mt_system_prompt),
        "labelling_method": "mt_context",
    },
    {
        "name": "mt_system_and_complex_labelling_context",
        "dataset": "mt",
        "preprocessing_fn": lambda x: add_system_prompt(x, mt_system_prompt),
        "labelling_method": "mt_extended_context",
    },
]


if __name__ == "__main__":
    for setting in settings:
        print(f"Labelling {setting['dataset']} with {setting['name']}")
        dataset = LabelledDataset.load_from(EVAL_DATASETS[setting["dataset"]])
        labelled_dataset = label_dataset(
            dataset=dataset.sample(max_samples),  # type: ignore
            model=labelling_model,
            preprocessing_fn=setting["preprocessing_fn"],
            labelling_method=setting["labelling_method"],
        )
        filename = f"{setting['name']}_labelled"
        if max_samples is not None:
            filename += f"_{max_samples}"
        filename += ".jsonl"
        labelled_dataset.save_to(
            out_dir / filename,
            overwrite=True,
        )
