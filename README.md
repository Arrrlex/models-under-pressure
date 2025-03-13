# Models Under Pressure

## Generating Eval Datasets

Run the files `anthropic_dataset.py`, `mt_dataset.py`, `mts_dataset.py` and `toolace_dataset.py` from `src/models_under_pressure/eval_datasets/`.
That will create the corresponding dataset files (raw and balanced) in the evals directory (see config.py).

Run the script `label_distribution.py` from `src/models_under_pressure/scripts/` to see number of high-stakes/low-stakes/ambiguous
for all eval datasets.
