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
echo "Running Group: $ABL_GROUP in Pointmaze Ablations with Dual-Goal Representation Viscous Experiments"

# --- RUN COMMANDS ---
case $ABL_GROUP in
    "viscous_scale")
        python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.viscous_scale=0.0001 --agent.enable_fk_regularization=True --verbose=1
        python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.viscous_scale=0.0001 --agent.enable_fk_regularization=True --verbose=1

        python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.viscous_scale=0.01 --agent.enable_fk_regularization=True --verbose=1
        python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.viscous_scale=0.01 --agent.enable_fk_regularization=True --verbose=1

        python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.viscous_scale=0.1 --agent.enable_fk_regularization=True --verbose=1
        python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.viscous_scale=0.1 --agent.enable_fk_regularization=True --verbose=1
        ;;
    "num_walks")
        python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.num_walks=1 --agent.enable_fk_regularization=True --verbose=1
        python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.num_walks=1 --agent.enable_fk_regularization=True --verbose=1

        python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.num_walks=5 --agent.enable_fk_regularization=True --verbose=1
        python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.num_walks=5 --agent.enable_fk_regularization=True --verbose=1

        python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.num_walks=20 --agent.enable_fk_regularization=True --verbose=1
        python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.num_walks=20 --agent.enable_fk_regularization=True --verbose=1
        ;;
    "enable_viscous_metric")
        python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --verbose=1
        python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.viscous_scale=0.01 --verbose=1
        #python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.enable_viscous_metric=False --verbose=1
        #python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.enable_viscous_metric=False --verbose=1
        
        ;;
    "use_metric_only")
        #python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.use_metric_only=True --verbose=1
        
        python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.use_metric_only=True --verbose=1
        python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99 --agent.viscous_scale=0.01 --agent.enable_fk_regularization=True --verbose=1
        ;;
    *)
esac