#!/bin/bash
#SBATCH --job-name=pw-seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00        # INCREASED to 24h just in case 6 runs > 12h
#SBATCH --partition=smallgpu
#SBATCH -A lilly-rob1
#SBATCH --output=slurm-logs/slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=hviswan@purdue.edu

# --------- setup ---------
source define.sh 
export MUJOCO_GL=egl
cd ..

echo "Starting Sequence Job for Env: $TARGET_ENV"

# Definitions
COMMON_ARGS="--train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent.alpha=3.0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small"
AGENT_BASE="agents/gcivl/original_eikonal.py"
AGENT_OURS="agents/gcivl/original.py"

# --- 1. RUN BASELINE (Seeds 0, 1, 2) ---
echo "--- Starting BASELINE runs ---"
for seed in 0 1 2; do
    echo "Running Baseline | Env: $TARGET_ENV | Seed: $seed"
    python main.py --env_name=$TARGET_ENV --agent=$AGENT_BASE $COMMON_ARGS --seed $seed
done

# --- 2. RUN OURS (Seeds 0, 1, 2) ---
echo "--- Starting OURS runs ---"
for seed in 0 1 2; do
    echo "Running Ours | Env: $TARGET_ENV | Seed: $seed"
    python main.py --env_name=$TARGET_ENV --agent=$AGENT_OURS $COMMON_ARGS --seed $seed
done

echo "All 6 sequential runs completed for $TARGET_ENV"