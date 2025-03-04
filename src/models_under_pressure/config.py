import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from models_under_pressure.interfaces.dataset import LabelledDataset

DEFAULT_MODEL = "gpt-4o-mini"

BATCH_SIZE = 64

# DEVICE = "cuda"
DEVICE = "cpu"

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Paths to input files
INPUTS_DIR = DATA_DIR / "inputs"
METADATA_FIELDS_FILE = INPUTS_DIR / "metadata_fields.csv"
TOPICS_JSON = INPUTS_DIR / "topics.json"
SITUATION_FACTORS_JSON = INPUTS_DIR / "situation_factors.json"
FILTERED_SITUATION_FACTORS_CSV = INPUTS_DIR / "situation_topics.csv"
LABELING_RUBRIC_PATH = INPUTS_DIR / "labeling_rubric.md"
GENERATED_DATASET_TRAIN_TEST_SPLIT = (
    INPUTS_DIR / "generated_dataset_train_test_split.json"
)


# Paths to output files
RESULTS_DIR = DATA_DIR / "results"

# Evals files
EVALS_DIR = DATA_DIR / "evals"
ANTHROPIC_SAMPLES_CSV = EVALS_DIR / "anthropic_samples.csv"
TOOLACE_SAMPLES_CSV = EVALS_DIR / "toolace_samples.csv"
MT_SAMPLES_CSV = EVALS_DIR / "mt_samples.csv"

EVAL_DATASETS = {
    "anthropic": {
        "path": ANTHROPIC_SAMPLES_CSV,
        "field_mapping": {
            "id": "ids",
            "messages": "inputs",
            "high_stakes": "labels",
        },
    },
    "toolace": {
        "path": TOOLACE_SAMPLES_CSV,
        "field_mapping": {
            "inputs": "inputs",
            "labels": "labels",
            "ids": "ids",
        },
    },
    "mt": {
        "path": MT_SAMPLES_CSV,
        "field_mapping": {
            "inputs": "inputs",
            "labels": "labels",
            "ids": "ids",
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
    num_combinations_for_prompts: int = 5
    combination_variation: bool = False  # If None, all factors are used

    sample_seperately: bool = False
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
        return self.run_dir / "prompts.jsonl"

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


@dataclass(frozen=True)
class HeatmapRunConfig:
    model_name: str
    layers: list[int]
    dataset_path: Path  # TODO Set default for this
    max_samples: int | None = None
    variation_types: tuple[str, ...] = tuple(VARIATION_TYPES)
    split_path: Path = GENERATED_DATASET_TRAIN_TEST_SPLIT


@dataclass(frozen=True)
class ProbeEvalRunConfig:
    layer: int
    max_samples: int | None = None
    variation_type: str | None = None
    variation_value: str | None = None
    dataset_path: Path = Path("data/results/prompts_28_02_25.jsonl")
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    @property
    def output_filename(self) -> str:
        return f"{self.dataset_path.stem}_{self.model_name.split('/')[-1]}_{self.variation_type}_fig2.json"


class GenerateActivationsConfig(BaseModel):
    dataset: LabelledDataset
    model_name: str
    layer: int

    output_dir: Path = DATA_DIR / "activations"

    @property
    def acts_output_file(self) -> Path:
        model_name_path_safe = self.model_name.replace("/", "_")
        return (
            self.output_dir
            / f"{model_name_path_safe}_{self.dataset.stable_hash}_{self.layer}.npz"
        )

    @property
    def attn_mask_output_file(self) -> Path:
        model_name_path_safe = self.model_name.replace("/", "_")
        return (
            self.output_dir
            / f"{model_name_path_safe}_{self.dataset.stable_hash}_{self.layer}_attn_mask.npz"
        )
