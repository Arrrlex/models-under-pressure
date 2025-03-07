# MTSamples dataset can be downloaded from https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?resource=download
# It should be saved to data/raw/mt_samples.csv

import pandas as pd
from datasets import load_dataset

from models_under_pressure.config import EVAL_DATASETS
from models_under_pressure.eval_datasets.label_dataset import label_dataset
from models_under_pressure.interfaces.dataset import Dataset


def load_mt_samples() -> Dataset:
    ds = load_dataset("NickyNicky/medical_mtsamples", split="train")
    df = pd.DataFrame(ds)  # type: ignore
    # Filter out rows with empty transcriptions
    df = df[df["transcription"].notna() & (df["transcription"] != "")]
    return Dataset.from_pandas(df, field_mapping={"transcription": "inputs"})


def main(num_samples: int, overwrite: bool = False):
    print(f"Loading {num_samples} samples")
    dataset = load_mt_samples()
    dataset = dataset.sample(num_samples)
    dataset = label_dataset(dataset)
    print(f"Saving the data to {EVAL_DATASETS['mt']}")
    dataset.save_to(EVAL_DATASETS["mt"], overwrite=overwrite)


if __name__ == "__main__":
    main(num_samples=100, overwrite=True)
