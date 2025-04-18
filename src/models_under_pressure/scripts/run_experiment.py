import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from models_under_pressure.baselines.baselines import run_baselines
from models_under_pressure.config import (
    CONFIG_DIR,
    TRAIN_DIR,
    ChooseLayerConfig,
    EvalRunConfig,
    HeatmapRunConfig,
    RunBaselinesConfig,
    global_settings,
)
from models_under_pressure.experiments.cross_validation import choose_best_layer_via_cv
from models_under_pressure.experiments.evaluate_probes import run_evaluation
from models_under_pressure.experiments.generate_heatmaps import generate_heatmaps
from models_under_pressure.utils import double_check_config


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="config",
    version_base=None,
)
def run_experiment(config: DictConfig):
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    train_data_path = TRAIN_DIR / config.train_data

    if config.experiment == "evaluate_probe":
        evaluate_probe_config = EvalRunConfig(
            model_name=config.model.name,
            dataset_path=train_data_path,
            dataset_filters=config.train_filters,
            layer=config.model.layer,
            probe_spec=config.probe,
            max_samples=config.max_samples,
            eval_datasets=list(config.eval_datasets.values()),
            compute_activations=config.compute_activations,
            validation_dataset=config.validation_dataset,
        )
        if global_settings.DOUBLE_CHECK_CONFIG:
            double_check_config(evaluate_probe_config)
        run_evaluation(evaluate_probe_config)

    if config.experiment == "generalisation_heatmap":
        heatmap_config = HeatmapRunConfig(
            layer=config.model.layer,
            model_name=config.model.name,
            dataset_path=train_data_path,
            max_samples=config.max_samples,
            variation_types=config.variation_types,
            probe_spec=config.probe,
        )
        if global_settings.DOUBLE_CHECK_CONFIG:
            double_check_config(heatmap_config)
        generate_heatmaps(heatmap_config)

    if config.experiment == "cv":
        choose_layer_config = ChooseLayerConfig(
            model_name=config.model.name,
            dataset_path=train_data_path,
            cv_folds=config.cv.folds,
            batch_size=config.batch_size,
            max_samples=config.max_samples,
            layers=config.cv.layers,
            probe_spec=config.probe,
        )
        if global_settings.DOUBLE_CHECK_CONFIG:
            double_check_config(choose_layer_config)
        choose_best_layer_via_cv(choose_layer_config)

    if config.experiment in ["run_baselines", "run_baseline"]:
        # run_baselines: Run the baseline with all prompts
        # run_baseline: Run the baseline only with the prompt selected in the config
        run_baselines_config = RunBaselinesConfig(
            model_name=config.model.name,
            dataset_path=train_data_path,
            baseline_prompts=config.baselines.prompts
            if config.experiment == "run_baselines"
            else [config.model.baseline_prompt],
            eval_datasets=config.eval_datasets,
            max_samples=config.max_samples,
            batch_size=config.batch_size,
        )

        if global_settings.DOUBLE_CHECK_CONFIG:
            double_check_config(run_baselines_config)
        run_baselines(run_baselines_config)


if __name__ == "__main__":
    run_experiment()  # type: ignore
