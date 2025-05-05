import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from models_under_pressure.baselines.baselines import run_baselines
from models_under_pressure.config import (
    CONFIG_DIR,
    PROJECT_ROOT,
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
from models_under_pressure.utils import AttrDict, double_check_config


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="config",
    version_base=None,
)
def run_experiment(config: DictConfig):
    config = AttrDict(OmegaConf.to_container(config, resolve=True, enum_to_str=True))  # type: ignore

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    train_data_path = TRAIN_DIR / config.train_data
    eval_datasets = {k: PROJECT_ROOT / v for k, v in config.eval_datasets.items()}

    if config.experiment == "evaluate_probe":
        evaluate_probe_config = EvalRunConfig(
            model_name=config.model.name,
            dataset_path=train_data_path,
            dataset_filters=config.train_filters,
            layer=config.model.layer,
            probe_spec=config.probe,
            max_samples=config.max_samples,
            eval_datasets=list(eval_datasets.values()),
            compute_activations=config.compute_activations,
            validation_dataset=config.validation_dataset,
        )
        double_check_config(
            evaluate_probe_config, double_check=global_settings.DOUBLE_CHECK_CONFIG
        )
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
        double_check_config(
            heatmap_config, double_check=global_settings.DOUBLE_CHECK_CONFIG
        )
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
        double_check_config(
            choose_layer_config, double_check=global_settings.DOUBLE_CHECK_CONFIG
        )
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
            eval_datasets=eval_datasets,
            max_samples=config.max_samples,
            batch_size=config.batch_size,
        )
        double_check_config(
            run_baselines_config, double_check=global_settings.DOUBLE_CHECK_CONFIG
        )
        run_baselines(run_baselines_config)

    if config.experiment == "data_efficiency":
        pass
        # Create a data efficiency config.
        # data_efficiency_config = DataEfficiencyConfig(
        #     model_name=config.model.name,
        #     layer=config.model.layer,
        #     dataset_path=train_data_path,
        #     dataset_sizes=config.data_efficiency.dataset_sizes,
        #     probes=[
        #         ProbeSpec(
        #             name=config.probe.name,
        #             hyperparams=config.probe.hyperparams,
        #         )
        #     ],
        #     compute_activations=config.compute_activations,
        #     eval_dataset_paths=list(eval_datasets.values()),
        # )

        # # Should be defined via a hydra run config file:
        # data_efficiency_finetune_config = DataEfficiencyBaselineConfig(
        #     model_name_or_path=config.model.name,
        #     num_classes=2,
        #     ClassifierModule=config.baseline.classifier_module,
        #     batch_size=config.baseline.dataloader.batch_size,
        #     shuffle=config.baseline.dataloader.shuffle,
        #     logger=hydra.utils.instantiate(
        #         config.baseline.logger
        #     ),  # TODO: should this live here or have its own config?
        #     Trainer=config.baseline.trainer,
        # )

        # # Check the config before running:
        # if global_settings.DOUBLE_CHECK_CONFIG:
        #     double_check_config(data_efficiency_config)

        # # Run the data efficiency experiment:
        # results = run_data_efficiency_experiment(data_efficiency_config)
        # baseline_results = run_data_efficiency_finetune_baseline_with_activations(
        #     data_efficiency_config, data_efficiency_finetune_config
        # )

        # # Process the results and plot the outputs:
        # plot_data_efficiency_results(results, baseline_results)


if __name__ == "__main__":
    run_experiment()  # type: ignore
