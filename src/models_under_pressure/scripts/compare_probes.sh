#! /bin/bash

# Exit on error
set -e

export DOUBLE_CHECK_CONFIG=false

# Compare all probes
for probe in mean softmax attention last max_of_rolling_mean max; do
    random_seed=$((RANDOM % 1000000))
    echo "Running with random seed $random_seed"
    uv run mup exp +experiment=evaluate_probe probe=$probe random_seed=$random_seed +id="${probe}_dev"
    uv run mup exp +experiment=evaluate_probe probe=$probe random_seed=$random_seed eval_datasets=test_balanced +id="${probe}_test"
done
