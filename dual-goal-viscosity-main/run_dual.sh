#!/bin/bash

# ==========================================
# 2. STATE TASKS (agents/gcivl/state/dual.py)
# ==========================================
echo "Running State GCIVL Dual experiments..."

# Humanoidmaze

# Antsoccer
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99

# Manipulation (Cube)
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# Manipulation (Scene)
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# Manipulation (Puzzle)
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995



# ==========================================
# 1. VISUAL TASKS (agents/gcivl/pixel/dual.py)
# ==========================================

# Pointmaze
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99

# Antmaze
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995