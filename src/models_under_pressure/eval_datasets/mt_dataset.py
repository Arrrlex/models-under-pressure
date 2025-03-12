# MTSamples dataset can be downloaded from https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?resource=download
# It should be saved to data/raw/mt_samples.csv

import pandas as pd
from datasets import load_dataset

from models_under_pressure.config import EVAL_DATASETS_BALANCED, EVAL_DATASETS_RAW
from models_under_pressure.eval_datasets.label_dataset import create_eval_dataset
from models_under_pressure.interfaces.dataset import Dataset


def load_mt_samples() -> Dataset:
    ds = load_dataset("NickyNicky/medical_mtsamples", split="train")
    df = pd.DataFrame(ds)  # type: ignore
    # Filter out rows with empty transcriptions
    df = df[df["transcription"].notna() & (df["transcription"] != "")]
    return Dataset.from_pandas(df, field_mapping={"transcription": "inputs"})


def create_mt_dataset(num_samples: int = 100, recompute: bool = False):
    dataset = load_mt_samples()
    dataset = dataset.sample(num_samples)
    return create_eval_dataset(
        dataset,
        raw_output_path=EVAL_DATASETS_RAW["mt"],
        balanced_output_path=EVAL_DATASETS_BALANCED["mt"],
        recompute=recompute,
    )


if __name__ == "__main__":
    create_mt_dataset(num_samples=20)
