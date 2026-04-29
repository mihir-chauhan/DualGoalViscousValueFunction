#!/bin/bash
#SBATCH --job-name=train-dgvisc1
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=smallgpu
#SBATCH -A lilly-rob1
#SBATCH --output=slurm-logs/slurm-%j-%x.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=hviswan@purdue.edu

# --------- setup ---------
source define.sh
export MUJOCO_GL=egl
cd ..

# EXPECTED VARIABLES: $ENV_GROUP, $SEED
echo "Running Group: $ENV_GROUP with Seed: $SEED"

# --- RUN COMMANDS ---
case $ENV_GROUP in
  "pointmaze")
    # Approx 8 Hours
    python main.py --env_name=pointmaze-medium-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=pointmaze-large-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=pointmaze-giant-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=pointmaze-teleport-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=pointmaze-giant-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --seed $SEED
    ;;

  "antmaze")
    # Approx 11 Hours
    python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=antmaze-teleport-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antmaze-medium-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antmaze-giant-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=antmaze-teleport-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antmaze-medium-explore-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antmaze-large-explore-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antmaze-teleport-explore-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    ;;

  "antsoccer")
    # Approx 4 Hours
    python main.py --env_name=antsoccer-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antsoccer-arena-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=antsoccer-medium-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    ;;

  "humanoidmaze")
    # Approx 6 Hours
    python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=humanoidmaze-medium-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=humanoidmaze-large-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995 --seed $SEED
    python main.py --env_name=humanoidmaze-giant-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995 --seed $SEED
    ;;

  "puzzle")
    # Approx 8 Hours
    python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=puzzle-4x5-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=puzzle-4x6-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=puzzle-3x3-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=puzzle-4x4-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=puzzle-4x5-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=puzzle-4x6-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    ;;

  "cube_scene")
    # Approx 10 Hours
    python main.py --env_name=cube-single-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=cube-double-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=cube-triple-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=cube-quadruple-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=scene-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=cube-triple-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    python main.py --env_name=cube-quadruple-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --seed $SEED
    ;;

  *)
    echo "Error: Unknown environment group $ENV_GROUP"
    exit 1
    ;;
esac

# --- AUTO-SUBMIT NEXT SEED ---
if [[ "$SEED" -lt 2 ]]; then
  NEXT_SEED=$((SEED + 1))
  echo ">>> Job for Seed $SEED finished. Auto-submitting Seed $NEXT_SEED <<<"
  sbatch --export=ALL,ENV_GROUP=$ENV_GROUP,SEED=$NEXT_SEED --job-name=${ENV_GROUP}_s${NEXT_SEED} submit_job.sh
else
  echo ">>> All seeds finished for group $ENV_GROUP <<<"
fi