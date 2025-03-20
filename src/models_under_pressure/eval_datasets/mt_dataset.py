# MTSamples dataset can be downloaded from https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?resource=download
# It should be saved to data/raw/mt_samples.csv

import pandas as pd
from datasets import load_dataset

from models_under_pressure.config import EVAL_DATASETS_BALANCED, EVAL_DATASETS_RAW
from models_under_pressure.eval_datasets.label_dataset import (
    create_eval_dataset,
    create_test_dataset,
)
from models_under_pressure.interfaces.dataset import Dataset


def load_mt_samples() -> Dataset:
    ds = load_dataset("NickyNicky/medical_mtsamples", split="train")
    df = pd.DataFrame(ds)  # type: ignore
    # Rename "Unnamed: 0" column to "ids" right after loading so we have fixed IDs
    df = df.rename(columns={"Unnamed: 0": "ids"})
    # Filter out rows with empty transcriptions
    df = df[df["transcription"].notna() & (df["transcription"] != "")]
    return Dataset.from_pandas(df, field_mapping={"transcription": "inputs"})


def create_mt_dataset(
    num_samples: int = 100, recompute: bool = False, split: str = "dev"
):
    dataset = load_mt_samples()
    dataset = dataset.sample(num_samples)

    if split == "dev":
        return create_eval_dataset(
            dataset,
            raw_output_path=EVAL_DATASETS_RAW["mt"],
            balanced_output_path=EVAL_DATASETS_BALANCED["mt"],
            recompute=recompute,
        )
    elif split == "test":
        return create_test_dataset(
            dataset,
            dataset_name="mt",
            recompute=recompute,
        )
    else:
        raise ValueError(f"Invalid split: {split}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use",
    )
    parser.add_argument(
        "--recompute",
        action="store_false",
        help="Recompute labels even if they already exist",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split to generate (dev or test)",
    )
    args = parser.parse_args()

    create_mt_dataset(
        num_samples=args.num_samples,
        recompute=args.recompute,
        split=args.split,
    )
