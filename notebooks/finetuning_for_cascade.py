from models_under_pressure.baselines.finetune import (
    get_finetuned_baseline_results,
)
from models_under_pressure.config import (
    RESULTS_DIR,
    SYNTHETIC_DATASET_PATH,
    EVAL_DATASETS,
    FinetuneBaselineConfig,
)

if __name__ == "__main__":
    # Should be defined via a hydra run config file:
    finetune_config = FinetuneBaselineConfig(
        # model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        # model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_name_or_path="google/gemma-3-1b-it",
        # model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        num_classes=2,
        ClassifierModule={  # set here to the default values
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "scheduler_params": None,
            "class_weights": None,
            "label_smoothing": 0.0,
        },
        batch_size=8,
        shuffle=True,
        logger=None,
        num_workers=25,
        Trainer={
            "max_epochs": 10,  # 20,
            # "accelerator": "gpu",
            "accelerator": "gpu",
            "devices": -1,
            "precision": "bf16-true",
            # "strategy": "deepspeed_stage_2_offload",
            "strategy": "fsdp",
            # "strategy": "ddp",
            "default_root_dir": "/home/ubuntu/models-under-pressure/.cache",
            "accumulate_grad_batches": 8,
        },
    )

    baseline_results = get_finetuned_baseline_results(
        finetune_config,
        train_dataset_path=SYNTHETIC_DATASET_PATH,
        # train_dataset_path=TRAIN_DIR / "original_doubled_unconfounded",
        # checkpoint_path="/home/ubuntu/models-under-pressure/.cache/lightning_logs/version_16/checkpoints/finetune-baselines-google/gemma-3-12b-it-epoch=01.ckpt",
        # eval_datasets=TEST_DATASETS,
        eval_datasets=EVAL_DATASETS,
        max_samples=None,
        compute_activations=True,
        use_validation_set=True,
        results_dir=RESULTS_DIR,
    )
    print(baseline_results)

    # results = DataEfficiencyResults.model_validate(results_dict)
