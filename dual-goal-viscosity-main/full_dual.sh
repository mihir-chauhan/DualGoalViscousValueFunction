

# Stitch
python main.py --env_name=pointmaze-medium-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-giant-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.995
python main.py --env_name=pointmaze-teleport-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99



python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x5-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x6-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# Navigate

python main.py --env_name=antsoccer-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99

# Stitch
python main.py --env_name=antsoccer-arena-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antsoccer-medium-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99


# ==========================================
# MANIPULATION (Expectile: 0.7, Dim: 256)
# ==========================================



# Cube (Noisy)
python main.py --env_name=cube-single-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-triple-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-quadruple-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# Scene


# Puzzle (Play)


# Puzzle (Noisy)
python main.py --env_name=puzzle-3x3-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x5-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x6-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99


# ==========================================
# HUMANOIDMAZE (Expectile: 0.9, Dim: 256, Disc: 0.995)
# ==========================================

# Navigate
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995

# Stitch
python main.py --env_name=humanoidmaze-medium-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-giant-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995


# ==========================================
# ANTMAZE (Expectile: 0.9, Dim: 256)
# ==========================================

# Navigate
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99

python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antmaze-teleport-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99

# Stitch
python main.py --env_name=antmaze-medium-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antmaze-teleport-stitch-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99

# Explore


python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-triple-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-quadruple-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99


python main.py --env_name=scene-noisy-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# ==========================================
# ANTSOCCER (Expectile: 0.9, Dim: 256)
# ==========================================
# Cube (Play)

python main.py --env_name=antmaze-medium-explore-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-explore-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-teleport-explore-v0 --agent=agents/gcivl/state/dual.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99


# ==========================================
# POINTMAZE (Expectile: 0.9, Dim: 64)
# ==========================================

# Navigate
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-giant-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.995
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99