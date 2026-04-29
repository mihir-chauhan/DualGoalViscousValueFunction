# Sim-to-Real for Dual-Goal Viscous Value Function

Offline GCIVL-Dual training (with the proposed Walk-on-Spheres / FK
regularizer in `agents/gcivl/state/dual.py`) on real-Franka waypoint demos
collected via the CQN-AS `simtoreal` pipeline.

```
simtoreal/
├── __init__.py
├── convert_waypoints.py   waypoints_*.json  →  OGBench-style npz
├── train_real.py          offline training of GCIVL-Dual on the npz
├── eval_real.py           load snapshot, deploy on real Franka via franky
├── run_compare.sh         end-to-end: convert → train (FK on / off) → eval
└── README.md
```

## End-to-end

You already have the demos — they were recorded with
`CQN-AS-main/simtoreal/record_waypoints.py` and saved as
`waypoints_NNNN.json` files. They contain `joints[7]` and `gripper_open`
per timestep at 10 Hz.

### 1. Convert waypoints to an OGBench-style state-only dataset

```bash
python -m simtoreal.convert_waypoints \
    --waypoint-dir /path/to/waypoints \
    --out-dir      ./datasets/franka_real \
    --val-split    0.1
```

Produces in `./datasets/franka_real/`:
- `franka_real.npz`        — train split (observations, actions, terminals, valids)
- `franka_real-val.npz`    — held-out split, same layout
- `action_stats.npz`       — `min/max` for de-normalising policy outputs
- `franka_real.meta.json`  — sanity counts (num demos, traj lengths, dims)

Observation = `[q0..q6, gripper_open]` (8-d). Action = `[Δq0..Δq6, gripper_target]`
(8-d), normalised to `[-1, 1]` by `action_stats` at load time.

### 2. Train both variants (proposed FK reg vs baseline)

`agents/gcivl/state/dual.py` is the file that holds the proposed
`enable_fk_regularization` branch (the stochastic Walk-on-Spheres viscous
HJB residual at `agents/gcivl/state/dual.py:273`). Toggling that flag is the
only difference between proposed and baseline.

```bash
# Proposed (FK ON)
python -m simtoreal.train_real \
    --dataset_dir ./datasets/franka_real \
    --save_dir    ./runs/franka_real_fk \
    --agent.enable_fk_regularization=True \
    --train_steps 200000

# Baseline (FK OFF)
python -m simtoreal.train_real \
    --dataset_dir ./datasets/franka_real \
    --save_dir    ./runs/franka_real_nofk \
    --agent.enable_fk_regularization=False \
    --train_steps 200000
```

Training is identical to `main.py` (same agent, same dataset class,
same loss); we just skip OGBench env construction and online evaluation
because we don't have a simulator on this side.

Each run writes:
- `params_<step>.pkl`     — Flax pickled snapshot
- `flags.json`            — full agent config; `eval_real.py` reads this
- `action_stats.npz`      — copied next to the snapshot for eval convenience
- `train.csv`, `val.csv`  — per-step loss logs

### 3. Evaluate on the real robot

```bash
python -m simtoreal.eval_real \
    --robot-ip       192.168.131.41 \
    --save-dir       ./runs/franka_real_fk \
    --restore-step   200000 \
    --goal-from-demo ./datasets/franka_real/franka_real-val.npz \
    --goal-demo-idx  0 \
    --num-episodes   5
```

The script connects via `franky`, reads `[q, gripper_open]` each step, queries
the agent for an action, de-normalises with `action_stats`, clips per-step
joint deltas to `--joint-delta-clip`, and commands a `JointWaypointMotion`.
Success is `||q - q_goal||_2 ≤ --success-joint-thresh` (radians, default
0.05 ≈ 3°).

`--goal-from-demo` extracts the final state of the chosen trajectory in any
OGBench-style npz; pass `--goal-joints '[..,..]'` to specify an explicit
8-vector instead.

### 4. Dry-run (no robot, no franky)

```bash
python -m simtoreal.eval_real --dry-run \
    --save-dir       ./runs/franka_real_fk \
    --restore-step   200000 \
    --goal-from-demo ./datasets/franka_real/franka_real-val.npz \
    --num-episodes   1
```

The dry-run robot integrates commanded deltas in software and prints them.
Useful for confirming policy outputs and shapes without touching hardware.

## One-liner: convert → train both → eval

```bash
bash simtoreal/run_compare.sh \
    /path/to/waypoints \
    ./datasets/franka_real \
    ./runs/franka_real \
    192.168.131.41
```

Pass an empty 4th argument to skip the real-robot eval and just train.

## Comparing FK vs no-FK

Both runs save `eval_results.json` after `eval_real.py` finishes. Compare
`success_rate` and the per-episode `min_joint_dist` to quantify the effect
of the viscous HJB regularizer on real-world generalisation. The FK branch
should:
1. produce a smoother value landscape (lower `validation/value/value_loss`),
2. close on the goal more reliably from off-distribution starts.

The CQN-AS sim-to-real plotting helpers (`plot_training.py`,
`compare_evals.py`) read the same csv/json layout and can be repointed at
these run directories without modification.
