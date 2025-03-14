import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch

DEFAULT_MODEL = "gpt-4o"

if torch.cuda.is_available():
    DEVICE: str = "cuda"
    BATCH_SIZE = 16
elif torch.backends.mps.is_available():
    DEVICE: str = "mps"
    BATCH_SIZE = 4
else:
    DEVICE: str = "cpu"
    BATCH_SIZE = 4

DATA_DIR = Path(__file__).parent.parent.parent / "data"

LOCAL_MODELS = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
}

MODEL_MAX_MEMORY = {
    "meta-llama/Llama-3.2-1B-Instruct": None,
    "meta-llama/Llama-3.1-8B-Instruct": None,
    "meta-llama/Llama-3.3-70B-Instruct": {1: "80GB", 3: "80GB"},
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
GENERATED_DATASET_PATH = OUTPUT_DIR / "prompts_13_03_25_gpt-4o.jsonl"
HEATMAPS_DIR = RESULTS_DIR / "generate_heatmaps"
EVALUATE_PROBES_DIR = RESULTS_DIR / "evaluate_probes"
AIS_DIR = RESULTS_DIR / "ais_evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"
PROBES_DIR = DATA_DIR / "probes"
GENERATED_DATASET = {
    "file_path": GENERATED_DATASET_PATH,
    "field_mapping": {
        "id": "ids",
        "prompt": "inputs",
        "high_stakes": "labels",
    },
}

# Evals files
USE_BALANCED_DATASETS = True
EVALS_DIR = DATA_DIR / "evals"
MANUAL_DATASET_PATH = EVALS_DIR / "manual.csv"

EVAL_DATASETS_RAW = {
    "anthropic": EVALS_DIR / "anthropic_samples.csv",
    "toolace": EVALS_DIR / "toolace_samples.csv",
    "mt": EVALS_DIR / "mt_samples.csv",
    "mts": EVALS_DIR / "mts_samples.csv",
}

EVAL_DATASETS_BALANCED = {
    "toolace": EVALS_DIR / "toolace_samples_balanced.csv",
    "anthropic": EVALS_DIR / "anthropic_samples_balanced.csv",
    "mt": EVALS_DIR / "mt_samples_balanced.csv",
    "mts": EVALS_DIR / "mts_samples_balanced.csv",
}

EVAL_DATASETS = EVAL_DATASETS_BALANCED if USE_BALANCED_DATASETS else EVAL_DATASETS_RAW

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


@dataclass(frozen=True)
class RunConfig:
    """
    Configuration for a dataset generation run.

    Args:
        num_situations_to_sample: How many pairs of high- and low-stakes situations to sample from the examples_situations.csv file.
        num_combinations_for_prompts: How many pairs of prompts to generate for each situation.
        max_concurrent_llm_calls: Maximum number of concurrent LLM calls.
        write_mode: Whether to overwrite or append to the output file.
        model: The model to use for the generation.
        train_frac: The fraction of the data to use for the training set.
    """

    num_situations_to_sample: int = 150
    num_combinations_for_prompts: int = 12
    max_concurrent_llm_calls: int = 50
    write_mode: Literal["overwrite", "append"] = "overwrite"
    model: str = "gpt-4o"
    train_frac: float = 0.8

    run_id: str = "test"

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.run_id

    @property
    def suffix(self) -> str:
        date_str = datetime.now().strftime("%d_%m_%y")
        return f"{date_str}_{self.model}"

    @property
    def prompts_file(self) -> Path:
        return self.run_dir / f"prompts_{self.suffix}.jsonl"

    @property
    def metadata_file(self) -> Path:
        return self.run_dir / "prompts_with_metadata.jsonl"

    @property
    def situations_file(self) -> Path:
        return self.run_dir / f"situations_{self.suffix}.jsonl"

    @property
    def variations_file(self) -> Path:
        return INPUTS_DIR / "prompt_variations.json"

    @property
    def random_state(self) -> int:
        return 32


with open(INPUTS_DIR / "prompt_variations.json") as f:
    VARIATION_TYPES = list(json.load(f).keys())


DEFAULT_GPU_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
DEFAULT_OTHER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass(frozen=True)
class HeatmapRunConfig:
    layers: list[int]
    model_name: str = DEFAULT_GPU_MODEL if "cuda" in DEVICE else DEFAULT_OTHER_MODEL
    dataset_path: Path = GENERATED_DATASET_PATH
    max_samples: int | None = None
    variation_types: tuple[str, ...] = tuple(VARIATION_TYPES)

    def output_filename(self, variation_type: str) -> str:
        return f"{self.dataset_path.stem}_{self.model_name.split('/')[-1]}_{variation_type}_heatmap.json"


@dataclass(frozen=True)
class EvalRunConfig:
    layer: int
    max_samples: int | None = None
    variation_type: str | None = None
    variation_value: str | None = None
    dataset_path: Path = GENERATED_DATASET_PATH
    model_name: str = DEFAULT_GPU_MODEL if "cuda" in DEVICE else DEFAULT_OTHER_MODEL

    @property
    def output_filename(self) -> str:
        return f"{self.dataset_path.stem}_{self.model_name.split('/')[-1]}_{self.variation_type}_fig2.json"

    @property
    def random_seed(self) -> int:
        return 32


@dataclass(frozen=True)
class SafetyRunConfig:
    layer: int
    max_samples: int | None = None
    variation_type: str | None = None
    variation_value: str | None = None
    dataset_path: Path = GENERATED_DATASET_PATH
    model_name: str = DEFAULT_GPU_MODEL if "cuda" in DEVICE else DEFAULT_OTHER_MODEL

    @property
    def output_filename(self) -> str:
        return f"{self.dataset_path.stem}_{self.model_name.split('/')[-1]}_{self.variation_type}_fig1.json"
