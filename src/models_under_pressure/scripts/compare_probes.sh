#! /bin/bash

# Exit on error
set -e

export DOUBLE_CHECK_CONFIG=false

# Compare all probes
random_seed=543
mup exp +experiment=evaluate_probe probe=max  random_seed=$random_seed
mup exp +experiment=evaluate_probe probe=max  random_seed=$random_seed eval_datasets=test_balanced
mup exp +experiment=evaluate_probe probe=mean random_seed=$random_seed
