from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gpt-4o-mini"

# Paths to input files
INPUTS_DIR = Path(__file__).parent.parent.parent / "data" / "inputs"
METADATA_FIELDS_FILE = INPUTS_DIR / "metadata_fields.csv"
CATEGORY_JSON = INPUTS_DIR / "category_taxonomy.json"
FACTOR_JSON = INPUTS_DIR / "factor_taxonomy.json"
CATEGORY_EXAMPLES_CSV = INPUTS_DIR / "situation_examples_by_category.csv"
FACTOR_EXAMPLES_CSV = INPUTS_DIR / "situation_examples_by_factor.csv"

# Paths to output files
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"

# Evals files
EVALS_DIR = Path(__file__).parent.parent.parent / "data" / "evals"
ANTHROPIC_SAMPLES_CSV = EVALS_DIR / "anthropic_samples.csv"


@dataclass(frozen=True)
class RunConfig:
    num_situations_per_combination: int = 2
    num_prompts_per_situation: int = 2
    num_categories_to_sample: int | None = 2  # If None, all categories are used
    num_factors_to_sample: int | None = 2  # If None, all factors are used
    run_id: str = "debug"

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.run_id

    @property
    def full_examples_csv(self) -> Path:
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
