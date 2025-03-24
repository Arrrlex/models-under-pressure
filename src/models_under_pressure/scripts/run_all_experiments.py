import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pydantic import BaseModel

from models_under_pressure.config import (
    CONFIG_DIR,
    EVALUATE_PROBES_DIR,
    HEATMAPS_DIR,
    LOCAL_MODELS,
    TRAIN_DIR,
    ChooseLayerConfig,
    EvalRunConfig,
    HeatmapRunConfig,
)
from models_under_pressure.experiments.cross_validation import choose_best_layer_via_cv
from models_under_pressure.experiments.evaluate_probes import run_evaluation
from models_under_pressure.experiments.generate_heatmaps import generate_heatmap
from models_under_pressure.interfaces.activations import (
    Aggregator,
    Postprocessors,
    Preprocessors,
)


class RunAllExperimentsConfig(BaseModel):
    model_name: str
    training_data: Path
    batch_size: int
    cv_folds: int
    best_layer: int
    layers: list[int]
    max_samples: int | None
    experiments_to_run: list[str]
    probes: list[dict[str, str]]
    best_probe: dict[str, str]
    variation_types: tuple[str, ...]


model_name = LOCAL_MODELS["llama-70b"]


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="run_all_experiments_default.yaml",
    version_base=None,
)
def run_all_experiments(config: DictConfig):
    valid_experiments = [
        "cv",
        "compare_probes",
        "compare_best_probe_against_baseline",
        "generalisation_heatmap",
        "scaling_plot",
    ]

    assert (
        len(config.experiments_to_run) > 0
    ), "Must specify at least one experiment to run"

    assert any(
        experiment in config.experiments_to_run for experiment in valid_experiments
    ), f"Must specify at least one experiment from {valid_experiments} to run"

    if "cv" in config.experiments_to_run:
        print("Running CV...")
        choose_best_layer_via_cv(
            ChooseLayerConfig(
                model_name=config.model_name,
                dataset_spec={
                    "file_path_or_name": TRAIN_DIR / config.training_data,
                },
                cv_folds=config.cv_folds,
                preprocessor=config.best_probe["preprocessor"],
                postprocessor=config.best_probe["postprocessor"],
                batch_size=config.batch_size,
                max_samples=config.max_samples,
                layers=config.layers,
            )
        )

    if "compare_probes" in config.experiments_to_run:
        print("Running compare probes...")
        for probe in config.probes:
            eval_run_config = EvalRunConfig(
                model_name=config.model_name,
                dataset_path=TRAIN_DIR / config.training_data,
                layer=config.best_layer,
                probe_name=probe["name"],
                max_samples=config.max_samples,
            )
            eval_results = run_evaluation(
                eval_run_config,
                aggregator=Aggregator(
                    preprocessor=getattr(Preprocessors, probe["preprocessor"]),
                    postprocessor=getattr(Postprocessors, probe["postprocessor"]),
                ),
            )

            for eval_result in eval_results:
                eval_result.save_to(
                    EVALUATE_PROBES_DIR / eval_run_config.output_filename
                )

            # TODO: also save the probe coefficients (making sure they're
            # re-scaled to the activation space)

    if "compare_best_probe_against_baseline" in config.experiments_to_run:
        print("Running compare best probe against baseline...")
        # This recomputes the probe evaluation results for the best probe

        eval_run_config = EvalRunConfig(
            model_name=config.model_name,
            dataset_path=config.training_data,
            layer=config.best_layer,
            probe_name=config.best_probe["name"],
            max_samples=config.max_samples,
        )
        eval_results = run_evaluation(
            eval_run_config,
            aggregator=Aggregator(
                preprocessor=getattr(Preprocessors, config.best_probe["preprocessor"]),
                postprocessor=getattr(
                    Postprocessors, config.best_probe["postprocessor"]
                ),
            ),
        )

        for eval_result in eval_results:
            eval_result.save_to(EVALUATE_PROBES_DIR / eval_run_config.output_filename)

        # TODO: calculate & save the baselines

    if "generalisation_heatmap" in config.experiments_to_run:
        print("Running generalisation heatmap...")
        # Warning: this will fail if we choose a pytorch best_probe
        heatmap_config = HeatmapRunConfig(
            model_name=config.model_name,
            dataset_path=config.training_data,
            layers=[config.best_layer],
            max_samples=config.max_samples,
            variation_types=config.variation_types,
        )
        for variation_type in config.variation_types:
            heatmap_results = generate_heatmap(heatmap_config, variation_type)

            out_path = HEATMAPS_DIR / heatmap_config.output_filename(variation_type)

            json.dump(heatmap_results.to_dict(), open(out_path, "w"))

    if "scaling_plot" in config.experiments_to_run:
        pass
        # TODO: run the scaling plot experiment


if __name__ == "__main__":
    run_all_experiments()


# if __name__ == "__main__":
#     config = RunAllExperimentsConfig(
#         model_name=LOCAL_MODELS["llama-1b"],
#         training_data=TRAIN_DIR / "prompts_13_03_25_gpt-4o_filtered.jsonl",
#         batch_size=32,
#         cv_folds=5,
#         best_layer=5,
#         layers=[5, 6],
#         max_samples=20,
#         experiments_to_run=[
#             "cv",
#             "compare_probes",
#             "compare_best_probe_against_baseline",
#             "generalisation_heatmap",
#             "scaling_plot",
#         ],
#         probes=[
#             {
#                 "name": "sklearn_probe",
#                 "preprocessor": "mean",
#                 "postprocessor": "sigmoid",
#             },
#             {
#                 "name": "pytorch_per_token_probe",
#                 "preprocessor": "mean",
#                 "postprocessor": "sigmoid",
#             },
#         ],
#         best_probe={
#             "name": "sklearn_probe",
#             "preprocessor": "mean",
#             "postprocessor": "sigmoid",
#         },
#         variation_types=tuple(VARIATION_TYPES),
#     )

#     double_check_config(config)

#     run_all_experiments(config)
