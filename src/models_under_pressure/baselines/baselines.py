from models_under_pressure.baselines.continuation import (
    evaluate_likelihood_continuation_baseline,
    likelihood_continuation_prompts,
)
from models_under_pressure.config import RunBaselinesConfig
from models_under_pressure.model import LLMModel


def run_baselines(config: RunBaselinesConfig):
    model = LLMModel.load(config.model_name)
    for dataset_name, dataset_path in config.eval_datasets.items():
        for prompt_config in config.baseline_prompts:
            results = evaluate_likelihood_continuation_baseline(
                model=model,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                max_samples=config.max_samples,
                batch_size=config.batch_size,
                prompt_config=likelihood_continuation_prompts[prompt_config],
            )

            print(f"Saving results to {config.output_path}")
            results.save_to(config.output_path)
