#!/bin/bash

# Only submit SEED 0.
# The job script will automatically submit SEED 1 when it finishes, 
# and SEED 1 will submit SEED 2.
seed=2

echo "Submitting INITIAL jobs (Seed $seed only) to avoid QOS limits..."

sbatch --export=ALL,ENV_GROUP=pointmaze,SEED=$seed --job-name=pm_s${seed} submit_job.sh
sbatch --export=ALL,ENV_GROUP=antmaze,SEED=$seed --job-name=am_s${seed} submit_job.sh
sbatch --export=ALL,ENV_GROUP=antsoccer,SEED=$seed --job-name=as_s${seed} submit_job.sh
sbatch --export=ALL,ENV_GROUP=humanoidmaze,SEED=$seed --job-name=hm_s${seed} submit_job.sh
sbatch --export=ALL,ENV_GROUP=puzzle,SEED=$seed --job-name=puz_s${seed} submit_job.sh
sbatch --export=ALL,ENV_GROUP=cube_scene,SEED=$seed --job-name=cs_s${seed} submit_job.sh

echo "Done. 6 jobs submitted."
echo "Check queue with: squeue -u $USER"