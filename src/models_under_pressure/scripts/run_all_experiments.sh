#! /bin/bash

# Exit on error
set -e

export DOUBLE_CHECK_CONFIG=false


# Compare all probes
for probe in sklearn_mean_agg_probe difference_of_means lda pytorch_per_token_probe pytorch_per_entry_probe_mean pytorch_attention_probe; do
    run-exp +experiment=evaluate_probe probe=$probe
done

# Exit script
exit 0


# Compare all models (scaling)
for model in llama-1b llama-3b llama-8b llama-70b gemma-1b gemma-12b gemma-27b; do
    run-exp +experiment=evaluate_probe model=$model
done

# Cross-validation
run-exp +experiment=cv model=llama-1b

# Generalisation heatmap
run-exp +experiment=generalisation_heatmap

# Baselines
for model in llama-1b gemma-1b; do
    run-exp +experiment=run_baselines model=$model
done
