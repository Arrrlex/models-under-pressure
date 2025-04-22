#! /bin/bash

# Exit on error
set -e

export DOUBLE_CHECK_CONFIG=false


# Compare all probes
# for probe in sklearn per_entry difference_of_means lda attention max max_of_sentence_means mean_of_top_k  mean_of_top_k mean max_of_rolling_mean last; do
for probe in mean_of_top_k; do
    run-exp +experiment=evaluate_probe probe=$probe ++probe.hyperparams.device=cuda
done

# # Compare all models (scaling)
# for model in llama-1b llama-3b llama-8b llama-70b gemma-1b gemma-12b gemma-27b; do
#     run-exp +experiment=evaluate_probe model=$model
# done

# # Cross-validation
# run-exp +experiment=cv model=llama-1b

# # Generalisation heatmap
# run-exp +experiment=generalisation_heatmap

# # Baselines
# for model in llama-1b gemma-1b; do
#     run-exp +experiment=run_baselines model=$model
# done
