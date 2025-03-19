from pathlib import Path

from pydantic import BaseModel

from models_under_pressure.config import (
    MANUAL_DATASET_PATH,
    ChooseLayerConfig,
    EvalRunConfig,
    HeatmapRunConfig,
)
from models_under_pressure.experiments.cross_validation import (
    choose_best_layer_via_cv,
)
from models_under_pressure.experiments.evaluate_on_manual import (
    run_evaluation_on_manual,
)
from models_under_pressure.experiments.evaluate_probes import run_evaluation
from models_under_pressure.experiments.generate_heatmaps import generate_heatmap
from models_under_pressure.experiments.safety_evaluation import run_safety_evaluation
from models_under_pressure.experiments.train_on_manual import main as train_on_manual


class RunAllExperimentsConfig(BaseModel):
    model: str
    training_data_path: Path
    max_samples: int


def run_all_experiments(config: RunAllExperimentsConfig):
    results = choose_best_layer_via_cv(
        ChooseLayerConfig(
            model_name=config.model,
            dataset_path=config.training_data_path,
            max_samples=config.max_samples,
            cv_folds=5,
            preprocessor="mean",
            postprocessor="sigmoid",
        )
    )

    layer = results.best_layer

    generate_heatmap(
        config=HeatmapRunConfig(
            layers=[layer],
            max_samples=config.max_samples,
            model_name=config.model,
        ),
        variation_type="prompt_style",
    )

    run_evaluation(
        variation_type=None,
        variation_value=None,
        max_samples=config.max_samples,
        layer=layer,
        dataset_path=config.training_data_path,
        model_name=config.model,
    )

    run_safety_evaluation(
        variation_type=None,
        variation_value=None,
        max_samples=config.max_samples,
        layer=layer,
        dataset_path=config.training_data_path,
        model_name=config.model,
    )

    run_evaluation_on_manual(
        variation_type=None,
        variation_value=None,
        max_samples=config.max_samples,
        layer=layer,
        train_dataset_path=config.training_data_path,
        manual_dataset_path=MANUAL_DATASET_PATH,
        model_name=config.model,
    )

    train_on_manual(
        config=EvalRunConfig(
            max_samples=config.max_samples,
            layer=layer,
            model_name=config.model,
        ),
        evaluation_type="standard",
        dataset_path=config.training_data_path,
        train_dataset_type="manual",
    )


if __name__ == "__main__":
    config = RunAllExperimentsConfig(
        model="llama-1b",
        training_data_path=Path("data/generated_dataset.jsonl"),
        max_samples=10,
    )

    run_all_experiments(config)
