from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from datasets import Dataset
import evaluate
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from models_under_pressure.config import (
    EVAL_DATASETS_BALANCED,
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
)
from models_under_pressure.dataset_utils import load_dataset, load_train_test
from models_under_pressure.interfaces.dataset import Input, to_dialogue


def tokenize(tokenizer: AutoTokenizer, inputs: Sequence[Input]) -> Dict[str, Any]:
    """Tokenize inputs using the model's chat template."""
    dialogues = [to_dialogue(input) for input in inputs]
    input_dicts = [[d.model_dump() for d in dialogue] for dialogue in dialogues]

    input_str = tokenizer.apply_chat_template(
        input_dicts,
        tokenize=False,  # Return string instead of tokens
        add_generation_prompt=False,  # Don't add final assistant prefix
    )

    token_dict = tokenizer(
        input_str,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Remove the first token (usually a special token) from input_ids and attention_mask
    for k, v in token_dict.items():
        if k in ["input_ids", "attention_mask"]:
            token_dict[k] = v[:, 1:]

    return token_dict


def evaluate_model(
    model: AutoModelForSequenceClassification,  # type: ignore
    tokenizer: AutoTokenizer,
    dataset_paths: Dict[str, Path],
) -> Dict[str, float]:
    """Evaluate the model's AUROC on multiple datasets.

    Args:
        model: The trained model to evaluate
        tokenizer: The tokenizer to use
        dataset_paths: Dictionary mapping dataset names to their paths

    Returns:
        Dictionary mapping dataset names to their AUROC scores
    """
    results = {}
    model.eval()  # type: ignore

    for dataset_name, dataset_path in dataset_paths.items():
        # Load dataset
        dataset = load_dataset(dataset_path)
        tokens = tokenize(tokenizer, dataset.inputs)

        # Convert tokens to tensors and move to GPU
        input_ids = tokens["input_ids"].to(model.device)  # type: ignore
        attention_mask = tokens["attention_mask"].to(model.device)  # type: ignore
        labels = torch.tensor([label.to_int() for label in dataset.labels])

        # Get predictions
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(input_ids))):
                outputs = model(  # type: ignore
                    input_ids=input_ids[i : i + 1],
                    attention_mask=attention_mask[i : i + 1],
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                predictions.append(probs.item())

        # Calculate AUROC
        auroc = roc_auc_score(labels, predictions)
        results[dataset_name] = auroc

    return results


def main(base_model: str, training_dataset_path: Path):
    # 1. Prepare dataset
    train, test = load_train_test(training_dataset_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert LabelledDataset instances to HuggingFace datasets
    train_dict = {
        "text": train.inputs,  # Pass Input objects directly
        "label": [label.to_int() for label in train.labels],
    }
    test_dict = {
        "text": test.inputs,  # Pass Input objects directly
        "label": [label.to_int() for label in test.labels],
    }

    train_ds = Dataset.from_dict(train_dict)
    test_ds = Dataset.from_dict(test_dict)

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        # Use our tokenize function for proper tokenization
        tokenized = tokenize(tokenizer, batch["text"])
        tokenized["labels"] = batch["label"]
        return tokenized

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["text", "label"])
    test_ds = test_ds.map(preprocess, batched=True, remove_columns=["text", "label"])

    # 2. Load base model and wrap with LoRA
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,  # rank
        lora_alpha=32,  # scaling
        lora_dropout=0.1,  # dropout
    )
    model = get_peft_model(model, lora_cfg)

    # 3. Training setup
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir="llama3-stakes-classifier",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=600,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 4. Train!
    trainer.train()

    # 5. Save adapters and merged model
    model.save_pretrained("llama3-stakes-lora-adapters")

    # 6. Evaluate on multiple datasets
    eval_datasets = {
        "synthetic": SYNTHETIC_DATASET_PATH,
        # Add more datasets here as needed
    }

    results = evaluate_model(model, tokenizer, eval_datasets)
    print("\nEvaluation Results:")
    for dataset_name, auroc in results.items():
        print(f"{dataset_name}: AUROC = {auroc:.4f}")

    return model


def load_and_evaluate(
    base_model: str,
    adapter_path: str,
    eval_datasets: Dict[str, Path],
) -> Dict[str, float]:
    """Load a saved model and evaluate it on multiple datasets."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model and adapters
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to("cuda")  # Move model to GPU
    model.eval()  # Set to evaluation mode

    # Evaluate
    results = evaluate_model(model, tokenizer, eval_datasets)
    print("\nEvaluation Results:")
    for dataset_name, auroc in results.items():
        print(f"{dataset_name}: AUROC = {auroc:.4f}")

    return results


if __name__ == "__main__":
    # To train a new model:
    # main(
    #     base_model=LOCAL_MODELS["llama-1b"],
    #     training_dataset_path=SYNTHETIC_DATASET_PATH,
    # )

    # To evaluate a saved model:
    load_and_evaluate(
        base_model=LOCAL_MODELS["llama-1b"],
        adapter_path="llama3-stakes-lora-adapters",
        eval_datasets=EVAL_DATASETS_BALANCED,
    )
