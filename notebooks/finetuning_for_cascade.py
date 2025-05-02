from models_under_pressure.config import (
    EVAL_DATASETS,
    RESULTS_DIR,
    SYNTHETIC_DATASET_PATH,
    DataEfficiencyBaselineConfig,
)
from models_under_pressure.finetune_baselines import get_finetuned_baseline_results

if __name__ == "__main__":
    # Should be defined via a hydra run config file:
    finetune_config = DataEfficiencyBaselineConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        # model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        # model_name_or_path="google/gemma-3-12b-it",
        # model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        num_classes=2,
        ClassifierModule={  # set here to the default values
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
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
            "devices": 1,
            "precision": "bf16-true",
            "default_root_dir": "/home/ubuntu/models-under-pressure/.cache",
            # "default_root_dir": "/Users/john/code/models-under-pressure/.cache",
            "accumulate_grad_batches": 4,
        },
    )

    baseline_results = get_finetuned_baseline_results(
        finetune_config,
        train_dataset_path=SYNTHETIC_DATASET_PATH,
        eval_datasets=EVAL_DATASETS,
        max_samples=10,
        compute_activations=True,
    )
    print(baseline_results)

    # Save the results:
    with open(RESULTS_DIR / "finetuning.jsonl", "a") as f:
        for result in baseline_results:
            f.write(result.model_dump_json() + "\n")

    # results = DataEfficiencyResults.model_validate(results_dict)
