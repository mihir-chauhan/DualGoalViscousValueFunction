"""
convert_waypoints.py — Turn the CQN-AS simtoreal waypoint .json demos into an
OGBench-style state-based npz dataset that the dual-goal-viscosity trainer can
consume.

Input  (one file per demo, produced by CQN-AS/simtoreal/record_waypoints.py):
    waypoints_NNNN.json
    {
        "hz": 10.0,
        "num_waypoints": K,
        "waypoints": [
            {"joints": [q0..q6], "gripper_open": true/false},
            ...
        ]
    }

Output (one combined .npz, OGBench-compatible):
    observations: (N, 8) float32   — [q0..q6, gripper_open]
    actions     : (N, 8) float32   — [Δq0..Δq6, gripper_target]
                                     The last action of each trajectory is a zero
                                     placeholder so that
                                     observations[t+1] == s' for every valid t.
    terminals   : (N,)  float32    — 1.0 at the last timestep of each demo, else 0
    goals       : (N, 8) float32   — final state of the trajectory containing t

Why this shape:
- The dual-goal-viscosity Dataset class is a "compact dataset": next_observations
  are inferred from observations[t+1] and `valids` masks the last step of each
  trajectory.  ogbench/utils.load_dataset(..., compact_dataset=True) does this
  same trick by setting valids = 1 - terminals_original and then merging
  terminals.  We pre-format directly in the compact form so the trainer can
  load the file with no special-casing.
- Observations include the gripper bit so the agent can reason about grasp
  state; actions include a target gripper bit (raw, not normalised) and seven
  joint deltas (radians).  Action stats (min/max) are saved alongside so the
  eval-time controller can map [-1, 1] policy outputs back to physical
  commands.

Usage
-----
    python -m simtoreal.convert_waypoints \
        --waypoint-dir /path/to/waypoints \
        --out-dir      ./datasets/franka_real \
        --val-split    0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


JOINT_DIM = 7
OBS_DIM = JOINT_DIM + 1  # joints + gripper bit
ACT_DIM = JOINT_DIM + 1  # joint deltas + gripper command


def _load_one(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single waypoints_*.json into (observations, actions).

    Returns:
        obs: (K, 8)  float32 — joint angles + gripper_open per recorded step
        act: (K, 8)  float32 — joint deltas to next step + next gripper command;
                              the final row is zero (no next step).
    """
    with open(path, "r") as f:
        data = json.load(f)
    wps = data["waypoints"]
    K = len(wps)
    if K < 2:
        raise ValueError(f"{path} has < 2 waypoints — skipping.")

    obs = np.zeros((K, OBS_DIM), dtype=np.float32)
    act = np.zeros((K, ACT_DIM), dtype=np.float32)
    for i, w in enumerate(wps):
        q = np.asarray(w["joints"], dtype=np.float32)
        g = 1.0 if w["gripper_open"] else 0.0
        obs[i, :JOINT_DIM] = q
        obs[i, JOINT_DIM] = g
    # Action[i] = (q[i+1] - q[i], gripper_target = gripper at step i+1).
    # Last action is a zero placeholder; it lives at a terminal step and will
    # be masked out by the `valids` field during training.
    act[:-1, :JOINT_DIM] = obs[1:, :JOINT_DIM] - obs[:-1, :JOINT_DIM]
    act[:-1, JOINT_DIM] = obs[1:, JOINT_DIM]
    return obs, act


def _build_dataset(files: List[Path]) -> dict:
    obs_list, act_list, term_list, goal_list = [], [], [], []
    traj_lengths = []
    for f in files:
        try:
            obs, act = _load_one(f)
        except ValueError as e:
            print(f"  [skip] {e}")
            continue
        K = obs.shape[0]
        terms = np.zeros((K,), dtype=np.float32)
        terms[-1] = 1.0
        # Goal = the *final* state of the trajectory, broadcast across all steps.
        # We only stash this as a sanity field; GCDataset re-samples goals at
        # train time using p_curgoal / p_trajgoal / p_randomgoal.
        goal = np.broadcast_to(obs[-1:], (K, OBS_DIM)).astype(np.float32)
        obs_list.append(obs)
        act_list.append(act)
        term_list.append(terms)
        goal_list.append(goal.copy())
        traj_lengths.append(K)
        print(f"  {f.name}: {K} steps")

    if not obs_list:
        raise RuntimeError("No usable demos found.")

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    terminals = np.concatenate(term_list, axis=0)
    goals = np.concatenate(goal_list, axis=0)
    # `valids` mirrors OGBench's compact-dataset convention: 1 everywhere
    # except the last step of each trajectory (where there is no next state
    # to bootstrap from).
    valids = 1.0 - terminals
    # Match OGBench's compact-dataset terminal convention: a 1 on the last
    # *and* the second-to-last step of each trajectory so that masks/rewards
    # don't try to bootstrap past the trajectory boundary.
    new_terminals = np.concatenate([terminals[1:], np.array([1.0], dtype=np.float32)])
    terminals_compact = np.minimum(terminals + new_terminals, 1.0).astype(np.float32)

    return dict(
        observations=observations,
        actions=actions,
        terminals=terminals_compact,
        valids=valids.astype(np.float32),
        goals=goals,
    ), traj_lengths


def _action_stats(actions: np.ndarray) -> dict:
    """Compute per-dim min/max for action normalisation.

    Pin the gripper dim to [0, 1] so eval-time de-normalisation always maps a
    policy output of +1 to 'open' and -1 to 'close', regardless of which
    gripper transitions happened in the demos.
    """
    a_min = np.min(actions, axis=0).astype(np.float32)
    a_max = np.max(actions, axis=0).astype(np.float32)
    a_min[-1] = 0.0
    a_max[-1] = 1.0
    # Guard against degenerate dims (constant column → min == max).
    span = a_max - a_min
    span = np.where(span < 1e-6, 1e-6, span)
    a_max = a_min + span
    return {"min": a_min, "max": a_max}


def main():
    p = argparse.ArgumentParser(
        description="Convert simtoreal waypoint demos → OGBench-style npz."
    )
    p.add_argument("--waypoint-dir", type=str, required=True,
                   help="Directory containing waypoints_*.json files.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Where to write franka_real.npz, franka_real-val.npz, "
                        "and action_stats.npz.")
    p.add_argument("--name", type=str, default="franka_real",
                   help="Dataset name (controls output file names).")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of demos held out for validation (>=1 demo).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    wp_dir = Path(args.waypoint_dir).expanduser()
    files = sorted(wp_dir.glob("waypoints_*.json"))
    if not files:
        raise FileNotFoundError(f"No waypoints_*.json files in {wp_dir}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * args.val_split)))
    val_idx = set(perm[:n_val].tolist())
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]
    print(f"Found {len(files)} demos -> train={len(train_files)}, val={len(val_files)}")

    print("\n[train]")
    train_data, train_lens = _build_dataset(train_files)
    print("\n[val]")
    val_data, val_lens = _build_dataset(val_files)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{args.name}.npz"
    val_path = out_dir / f"{args.name}-val.npz"
    np.savez_compressed(train_path, **train_data)
    np.savez_compressed(val_path, **val_data)
    print(f"\nSaved {train_path}")
    print(f"Saved {val_path}")

    stats = _action_stats(train_data["actions"])
    stats_path = out_dir / "action_stats.npz"
    np.savez(stats_path, **stats)
    print(f"Saved {stats_path}")
    print(f"  action min: {stats['min']}")
    print(f"  action max: {stats['max']}")

    meta = {
        "name": args.name,
        "obs_dim": int(OBS_DIM),
        "action_dim": int(ACT_DIM),
        "num_train_demos": len(train_files),
        "num_val_demos": len(val_files),
        "train_steps_total": int(train_data["observations"].shape[0]),
        "val_steps_total": int(val_data["observations"].shape[0]),
        "train_traj_lengths": train_lens,
        "val_traj_lengths": val_lens,
    }
    with open(out_dir / f"{args.name}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {out_dir / f'{args.name}.meta.json'}")


if __name__ == "__main__":
    main()
