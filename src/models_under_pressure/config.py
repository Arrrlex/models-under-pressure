import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings

from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.utils import generate_short_id


class GlobalSettings(BaseSettings):
    DEVICE: str = "auto"
    BATCH_SIZE: int = 4
    MODEL_MAX_MEMORY: dict[str, int | None] = Field(default_factory=dict)
    CACHE_DIR: str | None = None
    DEFAULT_MODEL: str = "gpt-4o"


global_settings = GlobalSettings()

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
ACTIVATIONS_DIR = DATA_DIR / "activations"

LOCAL_MODELS = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma-1b": "google/gemma-3-1b-it",
    "gemma-12b": "google/gemma-3-12b-it",
    "gemma-27b": "google/gemma-3-27b-it",
}

# Paths to input files
INPUTS_DIR = DATA_DIR / "inputs"
METADATA_FIELDS_FILE = INPUTS_DIR / "metadata_fields.csv"
TOPICS_JSON = INPUTS_DIR / "topics.json"
SITUATION_FACTORS_JSON = INPUTS_DIR / "situation_factors.json"
FILTERED_SITUATION_FACTORS_CSV = INPUTS_DIR / "situation_topics.csv"
LABELING_RUBRIC_PATH = INPUTS_DIR / "labeling_rubric.md"


# Paths to output files
RESULTS_DIR = DATA_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "outputs"
HEATMAPS_DIR = RESULTS_DIR / "generate_heatmaps"
EVALUATE_PROBES_DIR = RESULTS_DIR / "evaluate_probes"
AIS_DIR = RESULTS_DIR / "ais_evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"
PROBES_DIR = DATA_DIR / "probes"
BASELINE_RESULTS_FILE = PROBES_DIR / "continuation_baseline_results.jsonl"
BASELINE_RESULTS_FILE_TEST = PROBES_DIR / "continuation_baseline_results_test.jsonl"
TRAIN_DIR = DATA_DIR / "training"
EVALUATE_PROBES_DIR = RESULTS_DIR / "evaluate_probes"


# Training datasets

SYNTHETIC_DATASET_PATH = TRAIN_DIR / "prompts_25_03_25_gpt-4o.jsonl"

# Evals files
USE_BALANCED_DATASETS = True
EVALS_DIR = DATA_DIR / "evals" / "dev"
TEST_EVALS_DIR = DATA_DIR / "evals" / "test"

EVAL_DATASETS_RAW = {
    "manual": EVALS_DIR / "manual_upsampled.csv",
    "anthropic": EVALS_DIR / "anthropic_samples.csv",
    "toolace": EVALS_DIR / "toolace_samples.csv",
    "mt": EVALS_DIR / "mt_samples.csv",
    "mts": EVALS_DIR / "mts_samples.csv",
}

EVAL_DATASETS_BALANCED = {
    "manual": EVALS_DIR / "manual_upsampled.csv",
    "anthropic": EVALS_DIR / "anthropic_samples_balanced.jsonl",
    "toolace": EVALS_DIR / "toolace_samples_balanced.jsonl",
    "mt": EVALS_DIR / "mt_samples_balanced.jsonl",
    "mts": EVALS_DIR / "mts_samples_balanced.jsonl",
}

TEST_DATASETS_RAW = {
    "manual": TEST_EVALS_DIR / "manual.csv",
    "anthropic": TEST_EVALS_DIR / "anthropic_samples.csv",
    "toolace": TEST_EVALS_DIR / "toolace_samples.csv",
    "mt": TEST_EVALS_DIR / "mt_samples_clean.jsonl",
    "mts": TEST_EVALS_DIR / "mts_samples.csv",
    "mental_health": TEST_EVALS_DIR / "mental_health.jsonl",
    "redteaming": TEST_EVALS_DIR / "aya_redteaming.jsonl",
}

TEST_DATASETS_BALANCED = {
    "manual": TEST_EVALS_DIR / "manual.csv",
    "anthropic": TEST_EVALS_DIR / "anthropic_samples_balanced.jsonl",
    "toolace": TEST_EVALS_DIR / "toolace_samples_balanced.jsonl",
    "mt": TEST_EVALS_DIR / "mt_samples_clean_balanced.jsonl",
    "mts": TEST_EVALS_DIR / "mts_samples_balanced.jsonl",
    "mental_health": TEST_EVALS_DIR / "mental_health_balanced.jsonl",
    "redteaming": TEST_EVALS_DIR / "aya_redteaming_balanced.csv",
}

EVAL_DATASETS = EVAL_DATASETS_BALANCED if USE_BALANCED_DATASETS else EVAL_DATASETS_RAW
TEST_DATASETS = TEST_DATASETS_BALANCED if USE_BALANCED_DATASETS else TEST_DATASETS_RAW

AIS_DATASETS = {
    "mmlu_sandbagging": {
        "file_path_or_name": EVALS_DIR / "mmlu_sandbagging_labelled_dataset.jsonl",
        "field_mapping": {
            "labels": "high_stakes",
            "is_sandbagging": "labels",
        },
    },
    "deception": {
        "file_path_or_name": EVALS_DIR / "deception_labelled_.csv",
        "field_mapping": {
            "labels": "high_stakes",
            "is_deceptive": "labels",
            "id": "ids",
        },
    },
}

OTHER_DATASETS = {
    "redteaming_en": TEST_EVALS_DIR / "language/english_aya_redteaming.jsonl",
    "redteaming_fr": TEST_EVALS_DIR / "language/french_aya_redteaming.jsonl",
    "redteaming_hi": TEST_EVALS_DIR / "language/hindi_aya_redteaming.jsonl",
    "redteaming_es": TEST_EVALS_DIR / "language/spanish_aya_redteaming.jsonl",
    "deception_data": DATA_DIR / "evals/deception_data.yaml",
}


@dataclass(frozen=True)
class RunConfig:
    """

    num_situations_to_sample: How many situations to sample from the examples_situations.csv file.
    num_prompts_per_situation: How many prompts to generate for each situation. Each high or low stake prompt count as 1.
    num_situations_per_combination: How many situations to generate for each combination of topics and factors. Each high or low stake situation counts as 1.

    if num_situations_to_sample is 4 and num_situations_per_combination is 2, then 4*2 = 8 situations will be generated in the situations.jsonl file.
    Try to keep num_situations_per_combination as 2 to minimise weird behavior cause then LLM sometimesthinks of High and low stakes as seperate situations.
    The above is applicable for num_prompts_per_situation too.

    Based on the prompt variations, we need to decide num prompts per situation to sample.

    sample_seperately: if True sample from the topics and factors list directly rather than
    sampling from the examples_situations.csv file.

    """

    num_situations_per_combination: int = 2
    num_situations_to_sample: int = 150
    num_prompts_per_situation: int = 2
    num_topics_to_sample: int | None = 2  # If None, all topics are used
    num_factors_to_sample: int | None = 2
    num_combinations_for_prompts: int = 5
    combination_variation: bool = False  # If None, all factors are used

    sample_seperately: bool = False
    model: str = global_settings.DEFAULT_MODEL
    run_id: str = "test"
    train_frac: float = 0.8
    write_mode: str = "overwrite"
    max_concurrent_llm_calls: int = 50

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.run_id

    @property
    def situations_combined_csv(self) -> Path:
        return self.run_dir / "examples_situations.csv"

    @property
    def prompts_file(self) -> Path:
        date_str = datetime.now().strftime("%d_%m_%y")
        return self.run_dir / f"prompts_{date_str}_{self.model}.jsonl"

    @property
    def metadata_file(self) -> Path:
        return self.run_dir / "prompts_with_metadata.jsonl"

    @property
    def situations_file(self) -> Path:
        return self.run_dir / "situations.jsonl"

    @property
    def variations_file(self) -> Path:
        return INPUTS_DIR / "prompt_variations.json"

    @property
    def filtered_situations_file(self) -> Path:
        return self.run_dir / FILTERED_SITUATION_FACTORS_CSV


with open(INPUTS_DIR / "prompt_variations.json") as f:
    VARIATION_TYPES = list(json.load(f).keys())


DEFAULT_GPU_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
DEFAULT_OTHER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def val_train_data(cls, v: Path, info: ValidationInfo) -> Path:
    return TRAIN_DIR / v


def val_model(cls, v: str, info: ValidationInfo) -> str:
    return LOCAL_MODELS.get(v, v)


def val_models(cls, v: list[str], info: ValidationInfo) -> list[str]:
    return [LOCAL_MODELS.get(model, model) for model in v]


class HeatmapRunConfig(BaseModel):
    layer: int
    model_name: str
    dataset_path: Path
    max_samples: int | None
    variation_types: list[str]
    probe: ProbeSpec
    id: str = Field(default_factory=generate_short_id)
    timestamp: datetime = Field(default_factory=datetime.now)

    _val_model = field_validator("model_name", mode="after")(val_model)
    _val_train_data = field_validator("dataset_path", mode="after")(val_train_data)

    @property
    def output_path(self) -> Path:
        return HEATMAPS_DIR / f"results_{self.id}.jsonl"

    @property
    def intermediate_output_path(self) -> Path:
        return HEATMAPS_DIR / f"intermediate_results_{self.id}.jsonl"


class ChooseLayerConfig(BaseModel):
    model_name: str
    dataset_path: Path
    cv_folds: int
    batch_size: int
    probe: ProbeSpec
    max_samples: int | None = None
    layers: list[int] | None = None
    output_dir: Path = RESULTS_DIR / "cross_validation"
    layer_batch_size: int = 4

    _val_model = field_validator("model_name", mode="after")(val_model)
    _val_train_data = field_validator("dataset_path", mode="after")(val_train_data)

    @property
    def output_path(self) -> Path:
        return self.output_dir / "results.jsonl"

    @property
    def temp_output_path(self) -> Path:
        return self.output_dir / "temp_results.jsonl"


class EvalRunConfig(BaseModel):
    id: str = Field(default_factory=generate_short_id)
    layer: int
    probe: ProbeSpec
    use_test_set: bool = False
    hyper_params: dict[str, Any] | None = None
    max_samples: int | None = None
    variation_type: str | None = None
    variation_value: str | None = None
    dataset_path: Path = SYNTHETIC_DATASET_PATH
    model_name: str = (
        DEFAULT_GPU_MODEL if "cuda" in global_settings.DEVICE else DEFAULT_OTHER_MODEL
    )

    _val_model = field_validator("model_name", mode="after")(val_model)
    _val_train_data = field_validator("dataset_path", mode="after")(val_train_data)

    @property
    def output_filename(self) -> str:
        if self.use_test_set:
            return f"results_{self.id}_test.jsonl"
        else:
            return f"results_{self.id}.jsonl"

    @property
    def coefs_filename(self) -> str:
        stem = Path(self.output_filename).stem
        return f"{stem}_coefs.json"


class CompareProbesConfig(BaseModel):
    model_name: str
    dataset_path: Path
    layer: int
    max_samples: int | None
    use_test_set: bool
    probes: list[ProbeSpec]

    _val_model = field_validator("model_name", mode="after")(val_model)
    _val_dataset_path = field_validator("dataset_path", mode="after")(val_train_data)


class CompareProbeToBaselinesConfig(BaseModel):
    model_name: str
    dataset_path: Path
    layer: int
    max_samples: int | None
    use_test_set: bool
    probe: ProbeSpec
    baseline_models: list[str]
    baseline_prompts: list[str]
    batch_size: int = 4

    _val_model = field_validator("model_name", mode="after")(val_model)
    _val_dataset_path = field_validator("dataset_path", mode="after")(val_train_data)


class ScalingPlotConfig(BaseModel):
    models: list[str]
    layers: list[int]
    probe: ProbeSpec
    dataset_path: Path
    use_test_set: bool

    _val_models = field_validator("models", mode="after")(val_models)
    _val_dataset_path = field_validator("dataset_path", mode="after")(val_train_data)


class RunAllExperimentsConfig(BaseModel):
    random_seed: int = 42

    cross_validation: Optional[ChooseLayerConfig] = None
    compare_probes: Optional[CompareProbesConfig] = None
    compare_probe_to_baselines: Optional[CompareProbeToBaselinesConfig] = None
    generate_generalisation_heatmap: Optional[HeatmapRunConfig] = None
    scaling_plot: Optional[ScalingPlotConfig] = None
