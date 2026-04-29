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

echo "Running job type: $JOB_TYPE"
source define.sh
export MUJOCO_GL=egl
cd ..
case $JOB_TYPE in

  "orig_nofk")
    # VIB Agent: Pointmaze settings + FK Regularization
    python main.py \
      --env_name=pointmaze-large-navigate-v0 \
      --agent=agents/gcivl/original.py \
      --agent.alpha=10.0 \
      --agent.discount=0.99 \
      --agent.enable_fk_regularization=False
    ;;

  "orig")
    # Original Agent: Pointmaze settings + FK Regularization
    python main.py \
      --env_name=pointmaze-large-navigate-v0 \
      --agent=agents/gcivl/original.py \
      --agent.alpha=10.0 \
      --agent.discount=0.99 \
      --agent.enable_fk_regularization=True
    ;;

  "vib_nofk")
    # Eikonal Agent: Adapted Humanoid config to Pointmaze params (dim=64, beta=0.003, gamma=0.99)
    python main.py \
      --env_name=pointmaze-large-navigate-v0 \
      --agent=agents/gcivl/state/vib.py \
      --agent.alpha=10.0 \
      --agent.beta=0.003 \
      --agent.goalrep_dim=64 \
      --agent.discount=0.99 \
      --agent.enable_fk_regularization=False
    ;;
  "vib")
    # Eikonal Agent: Adapted Humanoid config to Pointmaze params (dim=64, beta=0.003, gamma=0.99)
    python main.py \
      --env_name=pointmaze-large-navigate-v0 \
      --agent=agents/gcivl/state/vib.py \
      --agent.alpha=10.0 \
      --agent.beta=0.003 \
      --agent.goalrep_dim=64 \
      --agent.discount=0.99 \
      --agent.enable_fk_regularization=True
    ;;

  "dual")
    # Dual Agent: Pointmaze settings (Bilinear, Expectile 0.9)
    # Note: enable_fk_regularization is usually implicit or not used in dual, 
    # but added here if the codebase supports it to match the prompt's requirement.
    python main.py \
      --env_name=pointmaze-large-navigate-v0 \
      --agent=agents/gcivl/state/dual.py \
      --agent.alpha=10.0 \
      --agent.rep_type=bilinear \
      --agent.rep_expectile=0.9 \
      --agent.goalrep_dim=64 \
      --agent.discount=0.99 \
      --agent.enable_fk_regularization=False
    ;;

   "eikonal")
    # Eikonal Agent: Adapted Humanoid config to Pointmaze params (dim=64, beta=0.003, gamma=0.99)
    python main.py \
      --env_name=pointmaze-large-navigate-v0 \
      --agent=agents/gcivl/state/eikonal_vib.py \
      --agent.alpha=10.0 \
      --agent.beta=0.003 \
      --agent.goalrep_dim=64 \
      --agent.discount=0.99 \
      --agent.enable_fk_regularization=True
    ;;

  *)
    echo "Error: Unknown JOB_TYPE '$JOB_TYPE'. Please specify vib, orig, eikonal, or dual."
    exit 1
    ;;

esac