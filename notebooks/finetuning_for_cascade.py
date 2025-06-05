from models_under_pressure.baselines.finetune import (
    get_finetuned_baseline_results,
    run_sanity_check,
)
from models_under_pressure.config import (
    EVAL_DATASETS,
    TEST_DATASETS,
    LOCAL_MODELS,
    RESULTS_DIR,
    SYNTHETIC_DATASET_PATH,
    FinetuneBaselineConfig,
)

if __name__ == "__main__":
    import pytorch_lightning as pl
    import torch

    # In a log message it was recommended to specify this precision for H200 SXM
    torch.set_float32_matmul_precision("high")

    for seed in [0, 1, 2]:
        pl.seed_everything(seed)

        debugging = False

        # Should be defined via a hydra run config file:
        finetune_config = FinetuneBaselineConfig(
            model_name_or_path=LOCAL_MODELS["gemma-12b"],
            # model_name_or_path=LOCAL_MODELS["llama-8b"],
            num_classes=2,
            ClassifierModule={  # set here to the default values
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "scheduler_params": {"name": "step", "step_size": 2, "gamma": 0.5},
                "class_weights": None,
                "label_smoothing": 0.0,
                "optimizer": "adamw8bit",
            },
            batch_size=2,
            test_batch_size=2,
            shuffle=True,
            logger=None,
            num_workers=25,
            Trainer={
                "max_epochs": 5,  # 20,
                # "accelerator": "gpu",
                "accelerator": "gpu",
                "devices": [0],
                "precision": "bf16-true",
                # "strategy": "deepspeed_stage_2_offload",
                # "strategy": "fsdp",
                "gradient_clip_val": 1.0,
                # "strategy": "ddp_find_unused_parameters_true",
                "default_root_dir": "/root/phil/models-under-pressure/.cache",
                "accumulate_grad_batches": 4,
            },
        )

        if debugging:
            run_sanity_check(
                finetune_config,
                dataset_path=EVAL_DATASETS["anthropic"],
                max_samples=300,
            )
        else:
            baseline_results = get_finetuned_baseline_results(
                finetune_config,
                train_dataset_path=SYNTHETIC_DATASET_PATH,
                # train_dataset_path=TRAIN_DIR / "original_doubled_unconfounded",
                eval_datasets=TEST_DATASETS,
                # eval_datasets=EVAL_DATASETS,
                # checkpoint_path="/home/ubuntu/models-under-pressure/.cache/lightning_logs/version_0/checkpoints/finetune-baselines-meta-llama/Llama-3.2-3B-Instruct-epoch=03.ckpt",
                max_samples=None,
                compute_activations=True,
                use_validation_set=True,
                results_dir=RESULTS_DIR,
                results_file=f"finetuning_gemma_12b_test_{seed}.jsonl",
            )
            if baseline_results:
                # print(baseline_results)

                for result in baseline_results:
                    print(f"Results on {result.dataset_name}:")
                    print(result.metrics)
