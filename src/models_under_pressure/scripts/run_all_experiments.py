import gc

import numpy as np
import torch

from models_under_pressure import pydra
from models_under_pressure.baselines.continuation import (
    evaluate_likelihood_continuation_baseline,
    likelihood_continuation_prompts,
)
from models_under_pressure.config import (
    BASELINE_RESULTS_FILE,
    BASELINE_RESULTS_FILE_TEST,
    CONFIG_DIR,
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    TEST_DATASETS,
    ChooseLayerConfig,
    EvalRunConfig,
    HeatmapRunConfig,
    RunAllExperimentsConfig,
)
from models_under_pressure.experiments.cross_validation import choose_best_layer_via_cv
from models_under_pressure.experiments.evaluate_probes import run_evaluation
from models_under_pressure.experiments.generate_heatmaps import generate_heatmaps
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.utils import double_check_config


@pydra.main(
    config_path=str(CONFIG_DIR),
    config_name="run_all_experiments_default",
    version_base=None,
)
def run_all_experiments(config: RunAllExperimentsConfig):
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    double_check_config(config)
    valid_experiments = [
        "cv",
        "compare_probes",
        "compare_best_probe_against_baseline",
        "generalisation_heatmap",
        "scaling_plot",
    ]

    if len(config.experiments_to_run) == 0:
        raise ValueError("Must specify at least one experiment to run")

    if invalid_experiments := set(config.experiments_to_run) - set(valid_experiments):
        raise ValueError(f"Invalid experiments: {invalid_experiments}")

    if "cv" in config.experiments_to_run:
        # TODO Consider hyper params
        print("Running CV...")
        choose_best_layer_via_cv(
            ChooseLayerConfig(
                model_name=config.model_name,
                dataset_spec={
                    "file_path_or_name": config.train_data,
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
                dataset_path=config.train_data,
                layer=config.best_layer,
                probe_spec=probe,
                max_samples=config.max_samples,
                use_test_set=config.use_test_set,
            )
            eval_results, _ = run_evaluation(eval_run_config)

            for eval_result in eval_results:
                print(
                    f"Saving results to {EVALUATE_PROBES_DIR / eval_run_config.output_filename}"
                )
                eval_result.save_to(
                    EVALUATE_PROBES_DIR / eval_run_config.output_filename
                )

            # Clean up memory
            del eval_results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # NOTE: For generating the plot, see notebooks/compare_best_probe_against_baseline.py
    if "compare_best_probe_against_baseline" in config.experiments_to_run:
        print("Running compare best probe against baseline...")
        # This recomputes the probe evaluation results for the best probe

        eval_run_config = EvalRunConfig(
            id="best_probe",
            model_name=config.model_name,
            dataset_path=config.train_data,
            layer=config.best_layer,
            probe_spec=config.best_probe,
            max_samples=config.max_samples,
            use_test_set=config.use_test_set,
        )

        eval_results, _ = run_evaluation(eval_run_config)

        for eval_result in eval_results:
            eval_result.save_to(EVALUATE_PROBES_DIR / eval_run_config.output_filename)

        # Clean up memory
        del eval_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Calculate & save the baselines
        for baseline_model in config.baseline_models:
            baseline_model_name = LOCAL_MODELS.get(baseline_model, baseline_model)
            model = LLMModel.load(baseline_model_name)
            if config.use_test_set:
                datasets = list(TEST_DATASETS.keys())
            else:
                datasets = list(EVAL_DATASETS.keys())
            for dataset_name in datasets:
                for prompt_config in config.baseline_prompts:
                    results = evaluate_likelihood_continuation_baseline(
                        model=model,
                        dataset_name=dataset_name,
                        max_samples=config.max_samples,
                        batch_size=config.batch_size,
                        use_test_set=config.use_test_set,
                        prompt_config=likelihood_continuation_prompts[prompt_config],
                    )

                    if config.use_test_set:
                        output_path = BASELINE_RESULTS_FILE_TEST
                    else:
                        output_path = BASELINE_RESULTS_FILE
                    print(f"Saving results to {output_path}")
                    results.save_to(output_path)

    if "generalisation_heatmap" in config.experiments_to_run:
        print("Running generalisation heatmap...")

        heatmap_config = HeatmapRunConfig(
            layer=config.best_layer,
            model_name=config.model_name,
            dataset_path=config.train_data,
            max_samples=config.max_samples,
            variation_types=config.variation_types,
            probe_spec=config.best_probe,
        )
        generate_heatmaps(heatmap_config)

    if "scaling_plot" in config.experiments_to_run:
        scaling_configs = [
            EvalRunConfig(
                layer=layer,
                model_name=LOCAL_MODELS[model],
                max_samples=None,
                dataset_path=config.train_data,
                probe_spec=config.scaling_plot.probe_spec,
                use_test_set=config.use_test_set,
            )
            for layer, model in zip(
                config.scaling_plot.scaling_layers, config.scaling_plot.scaling_models
            )
        ]

        for scaling_config in scaling_configs:
            print(
                f"Running evaluation for {scaling_config.id} and results will be saved to {EVALUATE_PROBES_DIR / scaling_config.output_filename}"
            )
            results, _ = run_evaluation(
                config=scaling_config,
            )

            print(
                f"Saving results for layer {scaling_config.layer} to {EVALUATE_PROBES_DIR / scaling_config.output_filename}"
            )
            for result in results:
                result.save_to(EVALUATE_PROBES_DIR / scaling_config.output_filename)


if __name__ == "__main__":
    run_all_experiments()  # type: ignore
