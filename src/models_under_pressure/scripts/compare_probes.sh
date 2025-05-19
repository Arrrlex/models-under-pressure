#! /bin/bash

# Exit on error
set -e

export DOUBLE_CHECK_CONFIG=false

# Compare all probes
for probe in mean softmax; do
    random_seed=$((RANDOM % 1000000))
    echo "Running with random seed $random_seed"
    mup exp +experiment=evaluate_probe probe=$probe random_seed=$random_seed +id="${probe}_dev"
    mup exp +experiment=evaluate_probe probe=$probe random_seed=$random_seed eval_datasets=test_balanced +id="${probe}_test"
done
