# Models Under Pressure

## Setup

In order to run this code:

1. Install uv and run `uv sync`
2. Create a cloudflare account and create an R2 bucket to store datasets & activations
3. Add a `.env` file to the project root with the following environment variables:
  ```
  OPENAI_API_KEY=
  OPEN_ROUTER_API_KEY=
  HF_TOKEN=
  R2_ACCESS_KEY_ID=
  R2_SECRET_ACCESS_KEY=
  R2_DATASETS_BUCKET=
  R2_ACTIVATIONS_BUCKET=
  R2_ACCOUNT_ID=
  ACTIVATIONS_DIR=
  HF_HOME=
  WANDB_API_KEY=
  ```

## Activations

In order to train or run inference with probes, you'll need to compute and store activations. You can do that using the `mup acts store` command. Here is an example:

```bash
uv run mup acts store --model 'meta-llama/Llama-3.2-1B-Instruct' --layer 11 --dataset data/training/prompts_4x/train.jsonl
```

This will compute activations, save them locally to `ACTIVATIONS_DIR`, and upload them to `R2_ACTIVATIONS_BUCKET`.

## Datasets

We contribute a new synthetic dataset we use for training, as well as slightly modified external datasets labelled for stakes we use for evaluation.

Our datasets can be found here:

| Dataset Name | Balanced | Raw |
|--------------|----------|-----|
| Training | [train](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/training/prompts_4x/train.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/training/prompts_4x/test.jsonl) | - |
| Anthropic HH | [dev](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/anthropic_balanced_apr_23.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/anthropic_test_balanced_apr_23.jsonl) | [dev](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/anthropic_raw_apr_23.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/anthropic_test_raw_apr_23.jsonl) |
| MT | [dev](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/mt_balanced_apr_30.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mt_test_balanced_apr_30.jsonl) | [dev](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/mt_raw_apr_30.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mt_test_raw_apr_30.jsonl) |
| MTS | [dev](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/mts_balanced_apr_22.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mts_test_balanced_apr_22.jsonl) | [dev](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/mts_raw_apr_22.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mts_test_raw_apr_22.jsonl) |
| Toolace | [dev](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/toolace_balanced_apr_22.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/toolace_test_balanced_apr_22.jsonl) | [dev](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/dev/toolace_raw_apr_22.jsonl); [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/toolace_test_raw_apr_22.jsonl) |
| Mental Health | [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mental_health_test_balanced_apr_22.jsonl) | [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/mental_health_test_raw_apr_22.jsonl) |
| Aya Redteaming | [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/aya_redteaming_balanced.csv) | [test](https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/aya_redteaming.jsonl) |


### Generating Dev Datasets for Evaluation

Run the files `anthropic_dataset.py`, `mt_dataset.py`, `mts_dataset.py` and `toolace_dataset.py` from `src/models_under_pressure/eval_datasets/`.
That will create the corresponding dataset files (raw and balanced) in the dev evals directory (see config.py).

Run the script `label_distribution.py` from `src/models_under_pressure/scripts/` to see number of high-stakes/low-stakes/ambiguous
for all eval datasets.

Warning: Creating further dev samples when test datasets are already present can lead to overlap between dev and test.
Ideally, first generate the full dev datasets and then create the test datasets to avoid overlap.

### Generating Test Datasets for Evaluation

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


### Deployment Context Datasets

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


## Running Experiments

### Probe Comparison
- Run `scripts/compare_probes.sh` to train probes and run prediction against all evaluation datasets
- Run `compare_probes_plot.py` to generate the bar chart

## Probe Calibration
Run `experiments/calibration.py`, modifying the paths in the `if __name == '__main__'` block to point to the relevant files generated by `compare_probes.sh`

## Probe Visualisation

To view the probe visualisation, run

### Training on Dev Split of Evaluation Datasets

- Run `experiments/dev_split_training.py` for the best probe with different settings of `dev_sample_usage`. The script computes results 5 times by default with the same settings.
  - Important: Set `gradient_accumulation_steps` to 1 in the config of the corresponding probe, since training data for this experiment can consist only of few samples and no learning occurs if number of batches is less than gradient accumulation steps.
- Run `figures/dev_split_training_plot.py` to generate the corresponding plot. Adjust file paths end of the file before.
  - If you want to include the line for the baseline, you can obtain the corresponding file from the cascade experiment.


### Data Efficiency Experiment

Code for running the data efficiency experiment is included in [experiments/data_efficiency.py](./src/models_under_pressure/experiments/data_efficiency.py). You can run that file directly after adjusting the configurations at the end of it and calling one of the functions defined in that file:

- Use the function `run_data_efficiency_experiment` to get results for different types of probes.
- Use the function `run_data_efficiency_finetune_baseline_with_activations` to compute results for the finetuned baselines. (Adjust config accordingly and run one baseline model at a time.)

Generate the plot by putting all results files into a single directory and calling the script [figures/plot_data_efficiency.py](./src/models_under_pressure/figures/plot_data_efficiency.py).

### Cascade Plot

- To generate finetuning results, run `notebooks/finetuning_for_cascade.py` (adjust the settings in that script depending on the model you want to finetune)
- To generate the other results, run `experiments/monitoring_cascade.py`. The corresponding configuration files can be found under `config/experiments/monitoring_cascade.yaml` and `config/experiments/monitoring_cascade/`. It has one part for computing the results and a second part to generate the plot based on the results.
  - Result generation: The script generates result files for the selected probe and the continuation baselines.
  - Plot generation: Make sure that all the relevant files are included in one directory. This typically involves moving the finetuned baseline results into the directory with the other results. Then run the analysis step of the script.
  - For generating the full cascade plot (appendix), make sure that in `analyze.yaml`, the `baseline_models` and `finetuned_baseline_models` selections are both set to null, so that all results are displayed. Also, you might want to tweak a few arguments of the plotting function such as reducing `y_lim`.

### Figure 1 Plot

- The script to generate this plot uses outputs from the cascade experiment. Run the cascade experiment to compute the full results (e.g. using null for model selections in `analyze.yaml`)
- Then run [figures/plot_method_comparison.py](./src/models_under_pressure/figures/plot_method_comparison.py)


## Computing Baselines

### Prompted Baselines

Run `uv run python src/models_under_pressure/scripts/run_experiment.py +experiment=run_baselines model=<MODEL>` (replacing `<MODEL>` by "llama-1b", "llama-70b", "gemma-1b" etc.) to generate the results of the respective prompted model on all dev datasets (make default for `eval_datasets` in `config/config.yaml` is set to `dev_balanced`) for all prompt templates. All results are written in JSONL format to a single results file.
