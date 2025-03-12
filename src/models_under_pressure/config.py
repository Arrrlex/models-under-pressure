import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch

DEFAULT_MODEL = "gpt-4o"

if torch.cuda.is_available():
    DEVICE: str = "cuda"
    BATCH_SIZE = 8
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
    "meta-llama/Llama-3.3-70B-Instruct": {1: "80GB", 2: "40GB", 3: "40GB"},
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
TRAIN_TEST_SPLIT = OUTPUT_DIR / "train_test_split.json"
GENERATED_DATASET_PATH = OUTPUT_DIR / "prompts_04_03_25_model-4o.jsonl"
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
USE_BALANCED_DATASETS = False
EVALS_DIR = DATA_DIR / "evals"

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
    num_combinations_for_prompts: int = 12
    max_concurrent_llm_calls: int = 50
    write_mode: Literal["overwrite", "append"] = "overwrite"
    model: str = "gpt-4o"

    run_id: str = "test"

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
        return self.run_dir / f"prompts_{date_str}_{DEFAULT_MODEL}.jsonl"

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
    split_path: Path = TRAIN_TEST_SPLIT

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
    split_path: Path = TRAIN_TEST_SPLIT

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
    split_path: Path = TRAIN_TEST_SPLIT

    @property
    def output_filename(self) -> str:
        return f"{self.dataset_path.stem}_{self.model_name.split('/')[-1]}_{self.variation_type}_fig1.json"
