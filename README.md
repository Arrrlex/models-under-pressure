# Models Under Pressure

## Dataset

We contribute 2 new datasets in this repository:

- Our synthetic dataset can be found at [data/training/prompts_25_03_25_gpt-4o.jsonl](./data/training/prompts_25_03_25_gpt-4o.jsonl)
- Our manual data can be found at [data/evals/test/manual.csv](./data/evals/test/manual.csv), with a GPT-4.5-generated "upsampled" version at [data/evals/dev/manual_upsampled.csv](./data/evals/dev/manual_upsampled.csv).

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


### Notes on Dataset Versions

Apr 15 versions (all based on calling parts of `modify_dataset.py`):

- ToolACE dev and test datasets: Based on previous dev dataset (raw version), modify the system prompt and relabel after adding system prompt.
- Anthropic dev and test datasets: Adding system prompt to each sample, otherwise no changes.
- MT dev and test datasets: Adding system prompt to each sample, removing cases where transcription length is less than description length, and adding more info to input.

Apr 16 versions (all based on calling parts of `modify_dataset.py`)

- MTS dev and test datasets: Parsing conversations (using strict mode), adding system prompt and relabelling.
- MT dev and test datasets: Adding system prompt to each sample, adding more info to input and relabelling.
- Anthropic dev and test datasets: Adding system prompt to each sample and relabelling.


## Deployment Context Datasets

Medical deployment dataset:

- Pair IDs up to 60 were generated using Gemini 2.5 Pro
- Additional pairs were created with GPT 4.5, giving the pairs from Gemini as examples
- Script `create_deployment_datasets.py` was used to convert into proper Dataset and relabel (which led to removal of many items)
