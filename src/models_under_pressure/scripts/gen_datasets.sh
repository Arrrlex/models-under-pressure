#!/bin/bash

python -m models_under_pressure.eval_datasets.anthropic_dataset
python -m models_under_pressure.eval_datasets.toolace_dataset
python -m models_under_pressure.eval_datasets.mt_dataset
python -m models_under_pressure.eval_datasets.mts_dataset
