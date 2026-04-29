"""
generate_demos.py — Synthetic pointmaze demos on a real-room maze spec.

This is a faithful port of OGBench's `data_gen_scripts/generate_locomaze.py`
pointmaze branch (oracle subgoal + Gaussian action noise + 0.2-scaled point
update), rewritten to consume a `MazeSpec` JSON instead of building a MuJoCo
env. Output matches the OGBench compact-dataset format that the
dual-goal-viscosity trainer already consumes.

Why synthetic: for pointmaze the oracle policy is just "head toward the next
cell on the BFS path" — no neural-net expert needed. This is exactly what
generate_locomaze.py does in its `if 'point' in env_name` branch.

Usage
-----
    python -m simtoreal_maze.generate_demos \
        --maze-json   simtoreal_maze/lab_room.json \
        --out-dir     ./datasets/maze_lab \
        --num-episodes 1000 \
        --noise        0.2 \
        --max-steps    500
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import trange

from simtoreal_maze.maze_def import MazeSpec


STEP_SCALE = 0.2  # matches PointEnv: x' = x + 0.2 * action  (action in [-1, 1])


def _oracle_subgoal(maze: MazeSpec, xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    """Return the (x, y) of the next cell centre along the BFS path.

    If the agent is already inside the goal cell, return the goal directly.
    """
    cur_cell = maze.xy_to_cell(xy)
    goal_cell = maze.xy_to_cell(goal_xy)
    if cur_cell == goal_cell:
        return goal_xy
    path = maze.bfs_path(cur_cell, goal_cell)
    if len(path) < 2:
        return goal_xy
    return maze.cell_center_xy(path[1])


def _episode(
    maze: MazeSpec,
    init_cell: Tuple[int, int],
    goal_cell: Tuple[int, int],
    max_steps: int,
    noise: float,
    rng: np.random.Generator,
    success_tol_m: float,
) -> dict:
    """Roll out one synthetic demo. Stops when goal is reached or max_steps."""
    pos = maze.cell_center_xy(init_cell).copy()
    # Match OGBench reset noise (uniform ±0.1 cell units) so demos don't all
    # start at exact cell centres.
    pos += rng.uniform(-0.1, 0.1, size=2).astype(np.float32) * maze.cell_size_m
    goal_xy = maze.cell_center_xy(goal_cell)

    obs_list, act_list, term_list = [], [], []
    for step in range(max_steps):
        # Oracle direction toward the next subgoal.
        sg = _oracle_subgoal(maze, pos, goal_xy)
        d = sg - pos
        n = float(np.linalg.norm(d))
        direction = d / (n + 1e-6)
        action = direction + rng.normal(0.0, noise, size=2).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs_list.append(pos.copy())
        act_list.append(action.copy())

        # Tentative new position.
        new_pos = pos + STEP_SCALE * action
        # Reject moves that try to enter a wall cell — clamp to old pos.
        if not maze.is_free(maze.xy_to_cell(new_pos)):
            new_pos = pos
        pos = new_pos.astype(np.float32)

        reached = float(np.linalg.norm(pos - goal_xy)) <= success_tol_m
        term_list.append(1.0 if reached else 0.0)
        if reached:
            break
    return {
        "observations": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(act_list, dtype=np.float32),
        "terminals": np.asarray(term_list, dtype=np.float32),
    }


def _to_compact(traj: dict) -> dict:
    """Mark the last two steps of the trajectory as terminal (OGBench compact
    convention) and return a copy ready to concatenate with other trajs."""
    K = traj["observations"].shape[0]
    if K < 2:
        return None  # too short to bootstrap
    terms = np.zeros((K,), dtype=np.float32)
    terms[-1] = 1.0
    new_terms = np.concatenate([terms[1:], np.array([1.0], dtype=np.float32)])
    terms_compact = np.minimum(terms + new_terms, 1.0).astype(np.float32)
    valids = (1.0 - terms).astype(np.float32)
    return {
        "observations": traj["observations"],
        "actions": traj["actions"],
        "terminals": terms_compact,
        "valids": valids,
    }


def _action_stats(actions: np.ndarray) -> dict:
    """Pointmaze actions already live in [-1, 1] so stats are trivial; we
    write them anyway so eval_spot.py can use the same de-normalisation
    pipeline as the Franka eval."""
    return {
        "min": np.full((2,), -1.0, dtype=np.float32),
        "max": np.full((2,), +1.0, dtype=np.float32),
    }


def main():
    p = argparse.ArgumentParser(description="Generate synthetic pointmaze demos for a real-room maze.")
    p.add_argument("--maze-json", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--name", type=str, default="maze_real")
    p.add_argument("--num-episodes", type=int, default=1000,
                   help="Total trajectories to generate (90/10 train/val split).")
    p.add_argument("--noise", type=float, default=0.2,
                   help="Gaussian std added to the oracle action (matches OGBench default).")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Hard cap on per-trajectory length.")
    p.add_argument("--success-tol", type=float, default=None,
                   help="World-frame metres within goal to count as reached. "
                        "Defaults to cell_size / 2.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-type", choices=["path", "navigate"], default="navigate",
                   help="'path' = fixed start_cell→goal_cell from the JSON. "
                        "'navigate' = sample random start (any free cell) + "
                        "random goal (vertex cells) per episode (OGBench default).")
    args = p.parse_args()

    maze = MazeSpec.from_json(args.maze_json)
    rng = np.random.default_rng(args.seed)
    success_tol = args.success_tol if args.success_tol is not None else 0.5 * maze.cell_size_m
    free_cells = maze.free_cells()
    vertex_cells = maze.vertex_cells() or free_cells

    # 90/10 split.
    n_train = int(round(args.num_episodes * 0.9))
    n_val = max(1, args.num_episodes - n_train)
    print(f"[gen] maze {maze.name} {maze.shape} cells, "
          f"cell_size={maze.cell_size_m} m, "
          f"{len(free_cells)} free cells, {len(vertex_cells)} vertex cells")
    print(f"[gen] train_episodes={n_train}, val_episodes={n_val}, "
          f"noise={args.noise}, success_tol={success_tol:.3f} m")

    train_parts, val_parts = [], []
    for split, n_eps, dest in [("train", n_train, train_parts),
                                ("val",   n_val,   val_parts)]:
        for ep in trange(n_eps, desc=f"[gen/{split}]"):
            if args.dataset_type == "path":
                init_cell = tuple(maze.start_cell)
                goal_cell = tuple(maze.goal_cell)
                # Light start randomisation so the trainer sees state coverage.
                if rng.random() < 0.5:
                    init_cell = free_cells[rng.integers(len(free_cells))]
            else:
                init_cell = free_cells[rng.integers(len(free_cells))]
                goal_cell = vertex_cells[rng.integers(len(vertex_cells))]
                while goal_cell == init_cell:
                    goal_cell = vertex_cells[rng.integers(len(vertex_cells))]

            traj = _episode(maze, init_cell, goal_cell, args.max_steps,
                            args.noise, rng, success_tol)
            compact = _to_compact(traj)
            if compact is not None:
                dest.append(compact)

    def _stack(parts):
        if not parts:
            raise RuntimeError("No usable trajectories generated.")
        keys = parts[0].keys()
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}

    train = _stack(train_parts)
    val = _stack(val_parts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{args.name}.npz"
    val_path = out_dir / f"{args.name}-val.npz"
    np.savez_compressed(train_path, **train)
    np.savez_compressed(val_path, **val)
    np.savez(out_dir / "action_stats.npz", **_action_stats(train["actions"]))

    # Persist the maze spec next to the dataset so eval_spot.py can find it.
    with open(args.maze_json) as f:
        maze_json = json.load(f)
    with open(out_dir / "maze.json", "w") as f:
        json.dump(maze_json, f, indent=2)

    meta = {
        "name": args.name,
        "obs_dim": 2,
        "action_dim": 2,
        "num_train_steps": int(train["observations"].shape[0]),
        "num_val_steps": int(val["observations"].shape[0]),
        "num_train_episodes": len(train_parts),
        "num_val_episodes": len(val_parts),
        "noise": args.noise,
        "max_steps": args.max_steps,
        "step_scale_m": float(STEP_SCALE),
        "cell_size_m": maze.cell_size_m,
    }
    with open(out_dir / f"{args.name}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[gen] wrote {train_path} ({meta['num_train_steps']} steps)")
    print(f"[gen] wrote {val_path}   ({meta['num_val_steps']} steps)")


if __name__ == "__main__":
    main()
