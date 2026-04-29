# Job 1: Easy Environment (Runs all 6 configs)
sbatch --export=ALL,TARGET_ENV=powderworld-easy-play-v0 run_seq_powder.sh

# Job 2: Medium Environment (Runs all 6 configs)
sbatch --export=ALL,TARGET_ENV=powderworld-medium-play-v0 run_seq_powder.sh

# Job 3: Hard Environment (Runs all 6 configs)
sbatch --export=ALL,TARGET_ENV=powderworld-hard-play-v0 run_seq_powder.sh