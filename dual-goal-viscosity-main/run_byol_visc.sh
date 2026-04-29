#!/bin/bash

# ==========================================
# 2. STATE TASKS (agents/gcivl/state/dual.py)
# ==========================================
echo "Running State GCIVL BYOL Visc experiments..."

python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.discount=0.99 --agent.goalrep_dim=256 
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/byol.py  --agent.alpha=10.0 --agent.discount=0.99 --agent.goalrep_dim=256 
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/byol.py  --agent.alpha=10.0  --agent.discount=0.99 --agent.goalrep_dim=256 
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/byol.py  --agent.alpha=10.0 --agent.discount=0.99 --agent.goalrep_dim=256 
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/byol.py  --agent.alpha=10.0 --agent.discount=0.995 --agent.goalrep_dim=256 
# Humanoidmaze

# Antsoccer
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/byol.py  --agent.alpha=10.0 --agent.discount=0.99 --agent.goalrep_dim=256 

# Manipulation (Cube)
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/byol.py  --agent.alpha=10.0 --agent.discount=0.99 --agent.goalrep_dim=256 
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/byol.py  --agent.alpha=10.0 --agent.discount=0.99 --agent.goalrep_dim=256 

# Manipulation (Scene)
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/byol.py  --agent.alpha=10.0 --agent.discount=0.99 --agent.goalrep_dim=256 