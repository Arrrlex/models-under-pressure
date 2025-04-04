"""
Script to sync datasets between local storage and R2 storage.
"""

from models_under_pressure.config import (
    AIS_DATASETS,
    EVAL_DATASETS_BALANCED,
    EVAL_DATASETS_RAW,
    OTHER_DATASETS,
    PROJECT_ROOT,
    SYNTHETIC_DATASET_PATH,
    TEST_DATASETS_BALANCED,
    TEST_DATASETS_RAW,
)
from models_under_pressure.r2 import (
    DATASETS_BUCKET,
    download_file,
    file_exists_in_bucket,
    upload_file,
)

# Paths to datasets to sync
ALL_DATASETS = [
    path.resolve().relative_to(PROJECT_ROOT)
    for path in [
        *EVAL_DATASETS_RAW.values(),
        *EVAL_DATASETS_BALANCED.values(),
        *TEST_DATASETS_RAW.values(),
        *TEST_DATASETS_BALANCED.values(),
        *[v["file_path_or_name"] for v in AIS_DATASETS.values()],
        SYNTHETIC_DATASET_PATH,
        *OTHER_DATASETS.values(),
    ]
]


def download_all_datasets():
    """Download all datasets from R2 storage that are not present locally."""

    assert isinstance(DATASETS_BUCKET, str)
    for local_path in ALL_DATASETS:
        key = str(local_path)
        if not local_path.exists():
            print(f"Downloading {key} to {local_path}")
            download_file(bucket_name=DATASETS_BUCKET, key=key, local_path=local_path)


def upload_datasets():
    """Upload all datasets to R2 storage that are not present in the bucket."""

    assert isinstance(DATASETS_BUCKET, str)
    for local_path in ALL_DATASETS:
        key = str(local_path)
        if local_path.exists():
            if file_exists_in_bucket(DATASETS_BUCKET, key):
                print(f"{key} already exists in bucket, skipping")
            else:
                print(f"Uploading {key}")
                upload_file(bucket_name=DATASETS_BUCKET, key=key, local_path=local_path)
        else:
            print(f"Local file {local_path} does not exist, skipping upload")


def sync_all_datasets():
    """Sync all datasets between R2 storage and local storage."""
    download_all_datasets()
    upload_datasets()


if __name__ == "__main__":
    sync_all_datasets()
