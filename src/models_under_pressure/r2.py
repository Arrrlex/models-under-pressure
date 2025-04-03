import os
from pathlib import Path

import boto3
import dotenv


dotenv.load_dotenv()

ACTIVATIONS_BUCKET = os.getenv("R2_ACTIVATIONS_BUCKET")
DATASETS_BUCKET = os.getenv("R2_DATASETS_BUCKET")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")


def get_r2_client():
    """Get an R2 client using boto3."""
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


def download_file(bucket_name: str, key: str, local_path: Path) -> bool:
    """
    Download a file from R2 storage to a local path.

    Args:
        bucket_name: Name of the R2 bucket
        key: Key of the file in the bucket
        local_path: Local path to save the file to

    Returns:
        bool: True if download was successful, False otherwise
    """
    r2_client = get_r2_client()
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {key} to {local_path}")
        r2_client.download_file(bucket_name, key, str(local_path))
        return True
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return False


def upload_file(bucket_name: str, key: str, local_path: Path) -> bool:
    """
    Upload a file to R2 storage.

    Args:
        bucket_name: Name of the R2 bucket
        key: Key to store the file under in the bucket
        local_path: Local path of the file to upload

    Returns:
        bool: True if upload was successful, False otherwise
    """
    r2_client = get_r2_client()
    try:
        print(f"Uploading {key} to {bucket_name}")
        r2_client.upload_file(str(local_path), bucket_name, key)
        return True
    except Exception as e:
        print(f"Failed to upload {key}: {e}")
        return False


def file_exists_in_bucket(bucket_name: str, key: str) -> bool:
    """
    Check if a file exists in the R2 bucket.

    Args:
        bucket_name: Name of the R2 bucket
        key: Key of the file to check

    Returns:
        bool: True if file exists, False otherwise
    """
    r2_client = get_r2_client()
    try:
        r2_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except Exception:
        return False
