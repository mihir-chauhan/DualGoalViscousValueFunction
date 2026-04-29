#!/bin/bash

# Only submit SEED 0.
# The job script will automatically submit SEED 1, which will submit SEED 2.
seed=1

echo "Submitting INITIAL jobs (Seed $seed only)..."

# Cube
sbatch --export=ALL,ENV_GROUP=cube_play,SEED=$seed --job-name=cube_play_s${seed} partitioned_jobs.sh
sbatch --export=ALL,ENV_GROUP=cube_noisy,SEED=$seed --job-name=cube_noisy_s${seed} partitioned_jobs.sh

# Puzzle
sbatch --export=ALL,ENV_GROUP=puz_play,SEED=$seed --job-name=puz_play_s${seed} partitioned_jobs.sh
sbatch --export=ALL,ENV_GROUP=puz_noisy,SEED=$seed --job-name=puz_noisy_s${seed} partitioned_jobs.sh


#sbatch --export=ALL,ENV_GROUP=pm_nav,SEED=$seed --job-name=pm_nav_s${seed} partitioned_jobs.sh

# AntMaze (Note: am_explore includes Scene)
sbatch --export=ALL,ENV_GROUP=am_nav,SEED=$seed --job-name=am_nav_s${seed} partitioned_jobs.sh
sbatch --export=ALL,ENV_GROUP=am_stitch,SEED=$seed --job-name=am_stitch_s${seed} partitioned_jobs.sh
sbatch --export=ALL,ENV_GROUP=am_explore,SEED=$seed --job-name=am_explore_s${seed} partitioned_jobs.sh

# AntSoccer
sbatch --export=ALL,ENV_GROUP=antsoccer,SEED=$seed --job-name=antsoccer_s${seed} partitioned_jobs.sh

# HumanoidMaze
#sbatch --export=ALL,ENV_GROUP=hm_nav,SEED=$seed --job-name=hm_nav_s${seed} partitioned_jobs.sh
#sbatch --export=ALL,ENV_GROUP=hm_stitch,SEED=$seed --job-name=hm_stitch_s${seed} partitioned_jobs.sh



# PointMaze
sbatch --export=ALL,ENV_GROUP=pm_stitch,SEED=$seed --job-name=pm_stitch_s${seed} partitioned_jobs.sh



echo "Done. 12 jobs submitted."