from pathlib import Path
from typing import List, Optional

from models_under_pressure.config import (
    TRAIN_DIR,
    EVAL_DATASETS,
    DataEfficiencyBaselineConfig,
)
from models_under_pressure.dataset_utils import load_dataset, load_train_test
from models_under_pressure.finetune_baselines import FinetunedClassifier
from models_under_pressure.interfaces.results import FinetunedBaselineResults


def get_finetuned_baseline_results(
    finetune_config: DataEfficiencyBaselineConfig,
    train_dataset_path: Path,
    eval_dataset_paths: List[Path],
    max_samples: Optional[int] = None,
    compute_activations: bool = True,
) -> List[FinetunedBaselineResults]:
    print("Loading train dataset")
    train_dataset, val_dataset = load_train_test(
        dataset_path=train_dataset_path,
        model_name=None,
        layer=None,
        compute_activations=compute_activations,
        n_per_class=max_samples // 2 if max_samples else None,
    )
    print("Training finetuned baseline...")
    finetune_baseline = FinetunedClassifier(finetune_config)

    # Train the finetune baseline:
    finetune_baseline.train(
        train_dataset,
        val_dataset=val_dataset,
    )

    print("\nLoading eval datasets")
    # We'll use the first eval dataset for the BaselineResults
    eval_results = []
    for eval_dataset_path in eval_dataset_paths:
        eval_dataset = load_dataset(
            dataset_path=eval_dataset_path,
            model_name=None,
            layer=None,
            compute_activations=compute_activations,
            n_per_class=max_samples // 2 if max_samples else None,
        )

        results = finetune_baseline.get_results(eval_dataset)

        # Convert tensors to lists for BaselineResults
        labels = (
            results.labels.cpu().numpy().tolist() if results.labels is not None else []
        )
        ground_truth = eval_dataset.labels_numpy().tolist()

        # Create BaselineResults instance
        baseline_results = FinetunedBaselineResults(
            ids=list(eval_dataset.ids),
            accuracy=results.accuracy(),
            labels=labels,
            scores=results.probits.tolist() if results.logits is not None else [],
            ground_truth=ground_truth,
            ground_truth_scale_labels=list(eval_dataset.other_fields["scale_labels"])
            if "scale_labels" in eval_dataset.other_fields
            else None,
            dataset_name=eval_dataset_path.stem,
            dataset_path=eval_dataset_path,
            model_name=finetune_config.model_name_or_path,
            max_samples=max_samples,
        )

        eval_results.append(baseline_results)

    return eval_results


if __name__ == "__main__":
    # Should be defined via a hydra run config file:
    finetune_config = DataEfficiencyBaselineConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        num_classes=2,
        ClassifierModule={  # set here to the default values
            "learning_rate": 1e-5,
            "weight_decay": 1.0,
            "scheduler_params": None,
            "class_weights": None,
            "label_smoothing": 0.0,
        },
        batch_size=4,
        shuffle=True,
        logger=None,
        Trainer={
            "max_epochs": 1,  # 20,
            "accelerator": "gpu",
            "devices": [0],
            "precision": "bf16-true",
            "default_root_dir": "/home/ubuntu/models-under-pressure/.cache",
            # "default_root_dir": "/Users/john/code/models-under-pressure/.cache",
            "accumulate_grad_batches": 4,
        },
    )

    baseline_results = get_finetuned_baseline_results(
        finetune_config,
        # train_dataset_path=SYNTHETIC_DATASET_PATH,
        train_dataset_path=TRAIN_DIR / "prompts_25_03_25_gpt-4o.jsonl",
        eval_dataset_paths=list(EVAL_DATASETS.values())[:2],
        max_samples=10,
        compute_activations=True,
    )
    print(baseline_results)

    # # Reload the results as a test:
    # with open(RESULTS_DIR / f"data_efficiency/results_{config.id}.jsonl", "r") as f:
    #     results_dict = json.loads(f.readlines()[-1])

    # results = DataEfficiencyResults.model_validate(results_dict)
