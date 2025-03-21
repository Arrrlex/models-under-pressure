# from pathlib import Path

# import numpy as np

# from models_under_pressure.config import (
#     AIS_DATASETS,
#     AIS_DIR,
#     DEFAULT_GPU_MODEL,
#     DEFAULT_OTHER_MODEL,
#     DEVICE,
#     SafetyRunConfig,
# )
# from models_under_pressure.experiments.dataset_splitting import (
#     load_filtered_train_dataset,
# )
# from models_under_pressure.experiments.evaluate_probes import (
#     ProbeEvaluationResults,
# )
# from models_under_pressure.experiments.train_probes import train_probes_and_save_results
# from models_under_pressure.interfaces.dataset import LabelledDataset
# from models_under_pressure.utils import double_check_config


# def run_safety_evaluation(
#     layer: int,
#     dataset_path: Path,
#     model_name: str = DEFAULT_GPU_MODEL if "cuda" in DEVICE else DEFAULT_OTHER_MODEL,
#     variation_type: str | None = None,
#     variation_value: str | None = None,
#     max_samples: int | None = None,
# ) -> ProbeEvaluationResults:
#     """Compute AUROCs for figure 1."""
#     train_dataset = load_filtered_train_dataset(
#         dataset_path,
#         variation_type,
#         variation_value,
#         max_samples,
#     )

#     # Load eval datasets
#     print("Loading eval datasets ...")
#     eval_datasets = {}

#     # Sandbagging dataset
#     # TODO Make it possible to evaluate on high-stakes label as well
#     sandbagging_dataset = LabelledDataset.load_from(
#         **AIS_DATASETS["mmlu_sandbagging"],
#     )
#     deception_dataset = LabelledDataset.load_from(
#         **AIS_DATASETS["deception"],
#     )
#     if max_samples is not None:
#         print("Subsampling the dataset ...")
#         indices_sandbagging = np.random.choice(
#             range(len(sandbagging_dataset.ids)),
#             size=max_samples,
#             replace=False,
#         )
#         indices_deception = np.random.choice(
#             range(len(deception_dataset.ids)),
#             size=max_samples,
#             replace=False,
#         )
#         sandbagging_dataset = sandbagging_dataset[list(indices_sandbagging)]  # type: ignore
#         deception_dataset = deception_dataset[list(indices_deception)]  # type: ignore
#     eval_datasets["Sandbagging"] = sandbagging_dataset
#     eval_datasets["Deception"] = deception_dataset

#     # results_dict = train_probes_and_save_results(
#     #     model_name=model_name,
#     #     train_dataset=train_dataset,
#     #     train_dataset_path=dataset_path,
#     #     eval_datasets=eval_datasets,
#     #     layer=layer,
#     #     output_dir=AIS_DIR,
#     # )
#     # metrics = []
#     # dataset_names = []
#     # for path, (_, results) in results_dict.items():
#     #     print(f"Metrics for {Path(path).stem}: {results.metrics}")
#     #     metrics.append(results)
#     #     dataset_names.append(Path(path).stem)

#     # results = ProbeEvaluationResults(
#     #     metrics=metrics,
#     #     train_dataset_path=str(dataset_path),
#     #     datasets=dataset_names,
#     #     model_name=model_name,
#     #     variation_type=variation_type,
#     #     variation_value=variation_value,
#     # )
#     # return results


# if __name__ == "__main__":
#     # Set random seed for reproducibility
#     RANDOM_SEED = 0
#     np.random.seed(RANDOM_SEED)

#     config = SafetyRunConfig(
#         max_samples=40,
#         layer=11,
#     )
#     double_check_config(config)
#     results = run_safety_evaluation(
#         variation_type=config.variation_type,
#         variation_value=config.variation_value,
#         max_samples=config.max_samples,
#         layer=config.layer,
#         dataset_path=config.dataset_path,
#         model_name=config.model_name,
#     )

#     results.save_to(AIS_DIR / config.output_filename)
