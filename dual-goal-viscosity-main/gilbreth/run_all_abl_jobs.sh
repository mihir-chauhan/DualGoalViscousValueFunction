#!/bin/bash

# Only submit SEED 0.
# The job script will automatically submit SEED 1, which will submit SEED 2.

echo "Submitting INITIAL ABLATION jobs (Seed 1 only)..."

# Cube
#sbatch --export=ALL,ABL_GROUP=viscous_scale --job-name=viscous_scale_s pointmaze_abl.sh
#sbatch --export=ALL,ABL_GROUP=num_walks --job-name=num_walks_s pointmaze_abl.sh

# Puzzle
sbatch --export=ALL,ABL_GROUP=enable_viscous_metric --job-name=enable_viscous_metric_s pointmaze_abl.sh
sbatch --export=ALL,ABL_GROUP=use_metric_only --job-name=use_metric_only_s pointmaze_abl.sh