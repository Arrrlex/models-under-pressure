import gc

import numpy as np
import torch

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
    CompareProbeToBaselinesConfig,
    CompareProbesConfig,
    EvalRunConfig,
    RunAllExperimentsConfig,
    ScalingPlotConfig,
)
from models_under_pressure.experiments.cross_validation import choose_best_layer_via_cv
from models_under_pressure.experiments.evaluate_probes import run_evaluation
from models_under_pressure.experiments.generate_heatmaps import generate_heatmaps
from models_under_pressure.model import LLMModel
from models_under_pressure.utils import double_check_config, pydra


@pydra.main(
    config_path=str(CONFIG_DIR),
    config_name="run_all_experiments_default",
    version_base=None,
)
def run_all_experiments(config: RunAllExperimentsConfig):
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    double_check_config(config)

    if cross_validation_config := config.cross_validation:
        # TODO Consider hyper params
        print("Running CV...")
        choose_best_layer_via_cv(cross_validation_config)
        clean_up_memory()

    if compare_probes_config := config.compare_probes:
        print("Running compare probes...")
        compare_probes(compare_probes_config)
        clean_up_memory()

    # NOTE: For generating the plot, see notebooks/compare_best_probe_against_baseline.py
    if probe_baselines_config := config.compare_probe_to_baselines:
        print("Running compare best probe against baseline...")
        # This recomputes the probe evaluation results for the best probe
        compare_probe_to_baselines(probe_baselines_config)
        clean_up_memory()

    if heatmap_config := config.generate_generalisation_heatmap:
        print("Running generalisation heatmap...")
        generate_heatmaps(heatmap_config)
        clean_up_memory()

    if scaling_plot_config := config.scaling_plot:
        print("Running scaling plot...")
        scaling_plot(scaling_plot_config)
        clean_up_memory()


def compare_probes(config: CompareProbesConfig):
    for probe in config.probes:
        eval_run_config = EvalRunConfig(
            model_name=config.model_name,
            dataset_path=config.dataset_path,
            layer=config.layer,
            probe=probe,
            max_samples=config.max_samples,
            use_test_set=config.use_test_set,
        )
        eval_results, _ = run_evaluation(eval_run_config)

        for eval_result in eval_results:
            print(
                f"Saving results to {EVALUATE_PROBES_DIR / eval_run_config.output_filename}"
            )
            eval_result.save_to(EVALUATE_PROBES_DIR / eval_run_config.output_filename)


def compare_probe_to_baselines(config: CompareProbeToBaselinesConfig):
    eval_run_config = EvalRunConfig(
        id="best_probe",
        model_name=config.model_name,
        dataset_path=config.dataset_path,
        layer=config.layer,
        probe=config.probe,
        max_samples=config.max_samples,
        use_test_set=config.use_test_set,
    )

    eval_results, _ = run_evaluation(eval_run_config)

    for eval_result in eval_results:
        eval_result.save_to(EVALUATE_PROBES_DIR / eval_run_config.output_filename)

    # Clean up memory
    del eval_results
    clean_up_memory()

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


def scaling_plot(config: ScalingPlotConfig):
    scaling_configs = [
        EvalRunConfig(
            layer=layer,
            model_name=model,
            max_samples=None,
            dataset_path=config.dataset_path,
            probe=config.probe,
            use_test_set=config.use_test_set,
        )
        for layer, model in zip(config.layers, config.models)
    ]

    for scaling_config in scaling_configs:
        output_path = EVALUATE_PROBES_DIR / scaling_config.output_filename
        results, _ = run_evaluation(config=scaling_config)

        print(f"Saving results for layer {scaling_config.layer} to {output_path}")
        for result in results:
            result.save_to(output_path)


def clean_up_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_all_experiments()  # type: ignore
