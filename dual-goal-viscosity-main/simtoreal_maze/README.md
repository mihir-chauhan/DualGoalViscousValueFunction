# Sim-to-Real Maze for Boston Dynamics Spot

State-only `(x, y)` pointmaze, ported to a real room. Same agent
(`agents/gcivl/state/dual.py`), same FK regularizer ablation, same training
loop — only the dataset and the eval driver are different.

```
simtoreal_maze/
├── __init__.py
├── maze_def.py             room → grid + cell↔world helpers + BFS
├── lab_room.example.json   sample room layout (replace with yours)
├── generate_demos.py       BFS oracle + noise → OGBench-style npz
├── train_maze.py           offline GCIVL-Dual training
├── eval_spot.py            Spot SDK driver (or --dry-run)
└── README.md
```

## How this maps to OGBench's pointmaze

| OGBench (sim)                | Real-room here                                  |
|------------------------------|-------------------------------------------------|
| Hand-coded `maze_map` (large/medium/giant) | Your `lab_room.json` `maze_map`           |
| `maze_unit=4.0` MuJoCo blocks | `cell_size_m`, real metres                      |
| `PointEnv.step` adds `0.2 * action` to qpos | Spot waypoint = `xy + 0.2 * action`     |
| `actor_fn = lambda ob: ob[-2:]` (oracle subgoal direction) | `_oracle_subgoal` via BFS over your grid |
| Random init / vertex-cell goals | Same — `--dataset-type navigate`            |

The trainer doesn't know any of this changed. The dataset still has 2-D
observations and 2-D actions in `[-1, 1]`, with the OGBench compact
terminals/valids convention.

## 1. Describe the room

Edit `lab_room.example.json` (or copy it to `simtoreal_maze/lab_room.json`):

- `cell_size_m`: how big each grid cell is in metres. 1.0 m is a good default
  for Spot — gives the policy ~5 sub-steps per cell.
- `origin_xy_m`: world `(x, y)` of the centre of cell `(0, 0)`. Pick a
  landmark in your room (e.g. the corner where Spot starts) and measure
  every wall relative to that point.
- `maze_map`: 2-D list, row 0 is the top (smallest y). 0 = free, 1 = wall.
  Ring the whole room with 1's so demos can never escape.
- `start_cell`, `goal_cell`: `[row, col]` indices.

If you want to sanity-check the spec, `python -c` it:

```python
from simtoreal_maze.maze_def import MazeSpec
m = MazeSpec.from_json("simtoreal_maze/lab_room.json")
print(m.shape, m.cell_size_m, m.free_cells())
print("start xy:", m.cell_center_xy(m.start_cell))
print("goal  xy:", m.cell_center_xy(m.goal_cell))
print("BFS:", m.bfs_path(m.start_cell, m.goal_cell))
```

## 2. Generate demos

```bash
python -m simtoreal_maze.generate_demos \
    --maze-json    simtoreal_maze/lab_room.json \
    --out-dir      ./datasets/maze_lab \
    --num-episodes 1000 \
    --noise        0.2 \
    --max-steps    500
```

This is a port of `data_gen_scripts/generate_locomaze.py` for pointmaze:
each step, the oracle action is the unit-direction toward the next BFS cell
plus Gaussian noise (std 0.2). 1000 episodes ≈ 100k–200k transitions, which
is what OGBench uses for the published medium/large pointmaze datasets.

Outputs in `./datasets/maze_lab/`:

- `maze_real.npz`, `maze_real-val.npz` — train / val
- `action_stats.npz` — `[-1, 1]` (trivial; saved for symmetry with the Franka pipeline)
- `maze.json` — copy of your room spec (`eval_spot.py` reads it from here)
- `maze_real.meta.json` — sanity counts

## 3. Train (FK on vs off — the proposed ablation)

```bash
# Proposed (FK regularizer ON)
python -m simtoreal_maze.train_maze \
    --dataset_dir ./datasets/maze_lab \
    --save_dir    ./runs/maze_lab_fk \
    --train_steps 500000 \
    --agent.enable_fk_regularization=True

# Baseline (FK regularizer OFF)
python -m simtoreal_maze.train_maze \
    --dataset_dir ./datasets/maze_lab \
    --save_dir    ./runs/maze_lab_nofk \
    --train_steps 500000 \
    --agent.enable_fk_regularization=False
```

Same agent and dataset class as `simtoreal/train_real.py`; only the npz
underneath changes. The trainer copies `maze.json` and `action_stats.npz`
into the run directory so eval can find them automatically.

## 4. Run on Spot

```bash
# Dry run first — make sure the policy actually closes on the goal in sim.
python -m simtoreal_maze.eval_spot --dry-run \
    --save-dir       ./runs/maze_lab_fk \
    --restore-step   500000 \
    --num-episodes   3 \
    --max-steps      300

# Real Spot
python -m simtoreal_maze.eval_spot \
    --save-dir     ./runs/maze_lab_fk \
    --restore-step 500000 \
    --hostname     192.168.80.3 \
    --username     <user> \
    --password     <pw> \
    --num-episodes 3 \
    --max-steps    300
```

Safety:
- The operator must hold the e-stop. We do *not* request it programmatically.
- We power on, blocking-stand, drive to start, then run the policy. On exit
  (success, max_steps, or KeyboardInterrupt) Spot sits and powers off.
- The script clamps any agent action that would step into a `1` cell to keep
  Spot inside the maze. Don't rely on that — keep your real-room walls
  matching `lab_room.json` to within `cell_size_m / 2` or the policy can
  graze them.
- `--frame-offset "[dx, dy]"` adds a constant world offset if you can't
  start Spot exactly at `origin_xy_m`.

## Comparison

After both runs, compare:

- `runs/maze_lab_fk/eval_spot_results.json`
- `runs/maze_lab_nofk/eval_spot_results.json`

Watch `success_rate` and the per-episode `min_dist_m`. The FK branch should
take cleaner paths (the viscous HJB regularizer enforces a smoother value
landscape, especially near corners), and recover better from off-corridor
states because the smoothed value extrapolates more reliably.

For training-time diagnostics, plot the same fields as the Franka runs:
`validation/value/value_loss`, `validation/value/fk_loss`,
`training/value/v_max - v_min`. Same knobs, same failure modes (see
`simtoreal/train.txt`).
