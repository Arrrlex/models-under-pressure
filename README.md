# Models Under Pressure

## Generating Dev Datasets for Evaluation

Run the files `anthropic_dataset.py`, `mt_dataset.py`, `mts_dataset.py` and `toolace_dataset.py` from `src/models_under_pressure/eval_datasets/`.
That will create the corresponding dataset files (raw and balanced) in the dev evals directory (see config.py).

Run the script `label_distribution.py` from `src/models_under_pressure/scripts/` to see number of high-stakes/low-stakes/ambiguous
for all eval datasets.

Warning: Creating further dev samples when test datasets are already present can lead to overlap between dev and test.
Ideally, first generate the full dev datasets and then create the test datasets to avoid overlap.

## Generating Test Datasets for Evaluation

Run the files `anthropic_dataset.py`, `mt_dataset.py`, `mts_dataset.py` and `toolace_dataset.py` from `src/models_under_pressure/eval_datasets/`,
using the `--split=test` argument.
That will create the corresponding dataset files (raw and balanced) in the test evals directory (see config.py).

After that, run the script `eval_dataset_split_check.py` to ensure that there is no overlap between dev and test datasets.
(Note that MT does have duplicates, so for that dataset you can expect some overlap by default.)
