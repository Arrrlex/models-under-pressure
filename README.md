# Models Under Pressure

## Datasets

Our datasets can be found here:

- Training data:
  - [train split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/training/prompts_4x/train.jsonl)
  - [test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/training/prompts_4x/test.jsonl)
- Eval datasets:
  - Anthropic HH:
    - [balanced dev split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/anthropic_balanced_apr_23.jsonl)
    - [raw dev split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/anthropic_raw_apr_23.jsonl)
    - [balanced test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/anthropic_test_balanced_apr_23.jsonl)
    - [raw test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/anthropic_test_raw_apr_23.jsonl)
  - MT:
    - [balanced dev split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/mt_balanced_apr_30.jsonl)
    - [raw dev split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/mt_raw_apr_30.jsonl)
    - [balanced test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mt_test_balanced_apr_30.jsonl)
    - [raw test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mt_test_raw_apr_30.jsonl)
  - MTS:
    - [balanced dev split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/mts_balanced_apr_22.jsonl)
    - [raw dev split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/mts_raw_apr_22.jsonl)
    - [balanced test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mts_test_balanced_apr_22.jsonl)
    - [raw test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mts_test_raw_apr_22.jsonl)
  - Toolace:
    - [balanced dev split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/toolace_balanced_apr_22.jsonl)
    - [raw dev split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/toolace_raw_apr_22.jsonl)
    - [balanced test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/toolace_test_balanced_apr_22.jsonl)
    - [raw test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/toolace_test_raw_apr_22.jsonl)
  - Mental Health:
    - [balanced test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mental_health_test_balanced_apr_22.jsonl)
    - [raw test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mental_health_test_raw_apr_22.jsonl)
  - Aya Redteaming:
    - [balanced test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/aya_redteaming_balanced.csv)
    - [raw test split](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/aya_redteaming.jsonl)

## Running Experiments

Training on dev split plot:

- Run `experiments/dev_split_training.py` for the best probe with different settings of `dev_sample_usage`. The script computes results 5 times by default with the same settings.
  - Important: Set `gradient_accumulation_steps` to 1 in the config of the corresponding probe, since training data for this experiment can consist only of few samples and no learning occurs if number of batches is less than gradient accumulation steps.
- Run `figures/dev_split_training_plot.py` to generate the corresponding plot. Adjust file paths end of the file before.
  - If you want to include the line for the baseline, you can obtain the corresponding file from the cascade experiment.


## Computing Baselines

### Prompted Baselines

Run `uv run python src/models_under_pressure/scripts/run_experiment.py +experiment=run_baselines model=<MODEL>` (replacing `<MODEL>` by "llama-1b", "llama-70b", "gemma-1b" etc.) to generate the results of the respective prompted model on all dev datasets (make default for `eval_datasets` in `config/config.yaml` is set to `dev_balanced`) for all prompt templates. All results are written in JSONL format to a single results file.


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

Apr 22 versions (all based on calling parts of `modify_dataset.py`)

- MTS dev and test datasets: Same as before with slightly modified system prompt (fixed typo and dropping "guest families" part).
- ToolACE dev and test datasets: Based on original dataset, modify the system prompt (only changing first sentence and removing a later confusing sentence) and relabel after adding system prompt.
- Aya Redteaming dataset (only test): Added a system prompt and relabelled.
- Mental health dataset (only test): Added a system prompt and relabelled.

Apr 23 version of Anthropic (dev and test): Remove the duplicate system prompt, relabelling again just in case.

Apr 30 version of MT (dev and test): Moved 350 samples from test raw to dev raw and resampled the balanced versions.


## Deployment Context Datasets

Medical deployment dataset:

- Pair IDs up to 60 were generated using Gemini 2.5 Pro
- Additional pairs were created with GPT 4.5, giving the pairs from Gemini as examples
- Script `create_deployment_datasets.py` was used to convert into proper Dataset and relabel (which led to removal of many items)

Software deployment dataset:

- All items generated with GPT 4.5
- Script `create_deployment_datasets.py` was used to convert into proper Dataset and relabel (which led to removal of many items)

Chatbot deployment dataset:

- All items generated with GPT 4.5
- Script `create_deployment_datasets.py` was used to convert into proper Dataset and relabel (which led to removal of many items)

Combined deployment dataset: Created by concatenating all previous datasets.
