#!/bin/bash

# Only submit SEED 0.
# The job script will automatically submit SEED 1, which will submit SEED 2.
seed=2

echo "Submitting INITIAL Eikonal jobs (Seed $seed only)..."
# Puzzle
sbatch --export=ALL,ENV_GROUP=puz_play,SEED=$seed --job-name=Eik_puz_play_s${seed} submit_eikonal.sh
sbatch --export=ALL,ENV_GROUP=puz_noisy,SEED=$seed --job-name=Eik_puz_noisy_s${seed} submit_eikonal.sh

# Cube
#sbatch --export=ALL,ENV_GROUP=cube_play,SEED=$seed --job-name=Eik_cube_play_s${seed} submit_eikonal.sh
sbatch --export=ALL,ENV_GROUP=cube_noisy,SEED=$seed --job-name=Eik_cube_noisy_s${seed} submit_eikonal.sh

# PointMaze
sbatch --export=ALL,ENV_GROUP=pm_stitch,SEED=$seed --job-name=Eik_pm_stitch_s${seed} submit_eikonal.sh
#sbatch --export=ALL,ENV_GROUP=pm_nav,SEED=$seed --job-name=Eik_pm_nav_s${seed} submit_eikonal.sh

# AntMaze (Note: am_explore includes Scene)
sbatch --export=ALL,ENV_GROUP=am_nav,SEED=$seed --job-name=Eik_am_nav_s${seed} submit_eikonal.sh
#sbatch --export=ALL,ENV_GROUP=am_stitch,SEED=$seed --job-name=Eik_am_stitch_s${seed} submit_eikonal.sh
sbatch --export=ALL,ENV_GROUP=am_explore,SEED=$seed --job-name=Eik_am_explore_s${seed} submit_eikonal.sh

# AntSoccer
sbatch --export=ALL,ENV_GROUP=antsoccer,SEED=$seed --job-name=Eik_antsoccer_s${seed} submit_eikonal.sh

# HumanoidMaze
sbatch --export=ALL,ENV_GROUP=hm_nav,SEED=$seed --job-name=Eik_hm_nav_s${seed} submit_eikonal.sh
#sbatch --export=ALL,ENV_GROUP=hm_stitch,SEED=$seed --job-name=Eik_hm_stitch_s${seed} submit_eikonal.sh



echo "Done. 12 Eikonal jobs submitted."