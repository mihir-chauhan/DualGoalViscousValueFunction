echo "Running State GCIVL VIB Visc experiments..."
export MUJOCO_GL=egl

python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.003 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.003 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.003 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99