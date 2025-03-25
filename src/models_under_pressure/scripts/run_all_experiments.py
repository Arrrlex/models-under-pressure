import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pydantic import BaseModel

from models_under_pressure.baselines.continuation import (
    evaluate_likelihood_continuation_baseline,
)
from models_under_pressure.config import (
    BASELINE_RESULTS_FILE,
    CONFIG_DIR,
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    HEATMAPS_DIR,
    LOCAL_MODELS,
    TEST_DATASETS,
    TRAIN_DIR,
    ChooseLayerConfig,
    EvalRunConfig,
    HeatmapRunConfig,
)
from models_under_pressure.experiments.cross_validation import choose_best_layer_via_cv
from models_under_pressure.experiments.evaluate_probes import run_evaluation
from models_under_pressure.experiments.generate_heatmaps import generate_heatmap
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.utils import double_check_config


class RunAllExperimentsConfig(BaseModel):
    model_name: str
    baseline_models: list[str]
    train_data: str
    batch_size: int
    cv_folds: int
    best_layer: int
    layers: list[int]
    max_samples: int | None
    experiments_to_run: list[str]
    probes: list[str]
    best_probe: str
    variation_types: tuple[str, ...]
    use_test_set: bool

    @property
    def train_data_path(self) -> Path:
        return TRAIN_DIR / self.train_data


@hydra.main(
    config_path=str(CONFIG_DIR),
    version_base=None,
)
def run_all_experiments(config: DictConfig):
    double_check_config(config)
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
                    "file_path_or_name": config.train_data_path,
                },
                cv_folds=config.cv_folds,
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
                dataset_path=config.train_data_path,
                layer=config.best_layer,
                probe_name=probe,
                max_samples=config.max_samples,
                use_test_set=config.use_test_set,
            )
            eval_results = run_evaluation(eval_run_config)

            for eval_result in eval_results:
                print(
                    f"Saving results to {EVALUATE_PROBES_DIR / eval_run_config.output_filename}"
                )
                eval_result.save_to(
                    EVALUATE_PROBES_DIR / eval_run_config.output_filename
                )

            # TODO: also save the probe coefficients (making sure they're
            # re-scaled to the activation space)

    # NOTE: For generating the plot, see notebooks/compare_best_probe_against_baseline.py
    if "compare_best_probe_against_baseline" in config.experiments_to_run:
        print("Running compare best probe against baseline...")
        # This recomputes the probe evaluation results for the best probe

        eval_run_config = EvalRunConfig(
            id="best_probe",
            model_name=config.model_name,
            dataset_path=config.train_data_path,
            layer=config.best_layer,
            probe_name=config.best_probe,
            max_samples=config.max_samples,
            use_test_set=config.use_test_set,
        )
        eval_results = run_evaluation(eval_run_config)

        for eval_result in eval_results:
            eval_result.save_to(EVALUATE_PROBES_DIR / eval_run_config.output_filename)

        # Calculate & save the baselines
        for baseline_model in config.baseline_models:
            baseline_model_name = LOCAL_MODELS.get(baseline_model, baseline_model)
            model = LLMModel.load(baseline_model_name)
            if config.use_test_set:
                datasets = list(TEST_DATASETS.keys())
            else:
                datasets = list(EVAL_DATASETS.keys())
            for dataset_name in datasets:
                results = evaluate_likelihood_continuation_baseline(
                    model=model,
                    dataset_name=dataset_name,
                    max_samples=config.max_samples,
                    batch_size=config.batch_size,
                    use_test_set=config.use_test_set,
                )

                output_path = BASELINE_RESULTS_FILE
                if config.use_test_set:
                    # Insert _test before the file extension
                    parts = str(output_path).rsplit(".", 1)
                    output_path = Path(f"{parts[0]}_test.{parts[1]}")
                print(f"Saving results to {output_path}")
                results.save_to(output_path)

    if "generalisation_heatmap" in config.experiments_to_run:
        print("Running generalisation heatmap...")
        # Warning: this will fail if we choose a pytorch best_probe
        heatmap_config = HeatmapRunConfig(
            model_name=config.model_name,
            dataset_path=config.train_data_path,
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
    run_all_experiments()  # type: ignore
