#! /bin/bash

# Exit on error
set -e

export DOUBLE_CHECK_CONFIG=false

export random_seed=3345

mup exp +experiment=evaluate_probe probe=mean eval_datasets=test_balanced random_seed=$random_seed
mup exp +experiment=evaluate_probe probe=softmax random_seed=$random_seed
mup exp +experiment=evaluate_probe probe=softmax eval_datasets=test_balanced random_seed=$random_seed
