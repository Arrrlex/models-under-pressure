import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from models_under_pressure.config import DATA_DIR, HF_DATASET_REPO

# Maps HF repo file path → local path relative to DATA_DIR
HF_TO_LOCAL: dict[str, str] = {
    # Training
    "training/train.jsonl": "training/prompts_4x/train.jsonl",
    "training/test.jsonl": "training/prompts_4x/test.jsonl",
    # Anthropic HH
    "anthropic_hh_balanced/validation.jsonl": "evals/dev/anthropic_hh_balanced.jsonl",
    "anthropic_hh_balanced/test.jsonl": "evals/test/anthropic_hh_balanced.jsonl",
    "anthropic_hh_raw/validation.jsonl": "evals/dev/anthropic_hh_raw.jsonl",
    "anthropic_hh_raw/test.jsonl": "evals/test/anthropic_hh_raw.jsonl",
    # MT
    "mt_balanced/validation.jsonl": "evals/dev/mt_balanced.jsonl",
    "mt_balanced/test.jsonl": "evals/test/mt_balanced.jsonl",
    "mt_raw/validation.jsonl": "evals/dev/mt_raw.jsonl",
    "mt_raw/test.jsonl": "evals/test/mt_raw.jsonl",
    # MTS
    "mts_balanced/validation.jsonl": "evals/dev/mts_balanced.jsonl",
    "mts_balanced/test.jsonl": "evals/test/mts_balanced.jsonl",
    "mts_raw/validation.jsonl": "evals/dev/mts_raw.jsonl",
    "mts_raw/test.jsonl": "evals/test/mts_raw.jsonl",
    # Toolace
    "toolace_balanced/validation.jsonl": "evals/dev/toolace_balanced.jsonl",
    "toolace_balanced/test.jsonl": "evals/test/toolace_balanced.jsonl",
    "toolace_raw/validation.jsonl": "evals/dev/toolace_raw.jsonl",
    "toolace_raw/test.jsonl": "evals/test/toolace_raw.jsonl",
    # Mental Health (test only)
    "mental_health_balanced/test.jsonl": "evals/test/mental_health_balanced.jsonl",
    "mental_health_raw/test.jsonl": "evals/test/mental_health_raw.jsonl",
    # Aya Redteaming (test only)
    "aya_redteaming_balanced/test.jsonl": "evals/test/aya_redteaming_balanced.jsonl",
    "aya_redteaming_raw/test.jsonl": "evals/test/aya_redteaming_raw.jsonl",
}


def download_all_datasets() -> None:
    """Download all datasets from HuggingFace to the expected local paths."""
    to_download: list[tuple[str, Path]] = []

    for hf_path, local_rel in HF_TO_LOCAL.items():
        local_path = DATA_DIR / local_rel
        if local_path.exists():
            continue
        to_download.append((hf_path, local_path))

    if not to_download:
        print("All datasets are already downloaded")
        return

    for hf_path, local_path in tqdm(to_download, desc="Downloading datasets"):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=hf_path,
            repo_type="dataset",
        )
        shutil.copy2(downloaded, local_path)
