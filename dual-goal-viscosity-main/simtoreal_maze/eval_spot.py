"""
eval_spot.py — Deploy a trained pointmaze GCIVL-Dual agent on Boston Dynamics
Spot.

Per step:
  1. Read Spot's odometry-frame body pose → world (x, y).
  2. Query the agent for an action a ∈ [-1, 1]^2.
  3. Convert to a target waypoint:  target_xy = current_xy + 0.2 * a
     (matches the OGBench PointEnv update used in the synthetic demos.)
  4. Send a `trajectory_command` to that target in the body / odom frame.
  5. Stop when ||xy - goal_xy|| <= success_tol_m or step budget exhausted.

State convention: world frame (x, y) in metres, *aligned with the maze.json
origin*. Decide once where (0, 0) is in your room and put Spot's `power_on`
pose at that point (or supply --frame-offset to compensate). The maze grid
extends in +x (right / "col") and +y (forward / "row").

The Spot client APIs are imported lazily so a `--dry-run` works on any
machine. Real eval needs `bosdyn-client` / `bosdyn-api` installed and a Spot
estop already acquired by the operator (we do not request the estop here —
that is intentional, follow Boston Dynamics' safety guidance).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents import agents
from utils.flax_utils import restore_agent
from ml_collections import ConfigDict
import jax
import jax.numpy as jnp

from simtoreal_maze.maze_def import MazeSpec


STEP_SCALE = 0.2  # must match generate_demos.py / PointEnv


# ---------------------------------------------------------------------------
# Agent reconstruction
# ---------------------------------------------------------------------------


def _load_flags_config(save_dir: Path) -> ConfigDict:
    flags_path = save_dir / "flags.json"
    if not flags_path.exists():
        raise FileNotFoundError(f"{flags_path} not found.")
    with open(flags_path) as f:
        run_flags = json.load(f)
    if "agent" in run_flags:
        agent_cfg = run_flags["agent"]
    else:
        agent_cfg = {k.split(".", 1)[1]: v for k, v in run_flags.items() if k.startswith("agent.")}
    return ConfigDict(agent_cfg)


def _build_and_restore(save_dir: Path, restore_step: int, seed: int):
    config = _load_flags_config(save_dir)
    agent_cls = agents[config["agent_name"]]
    ex_obs = np.zeros((1, 2), dtype=np.float32)
    ex_act = np.zeros((1, 2), dtype=np.float32)
    ex_goals = (
        np.zeros((1, config["goalrep_dim"]), dtype=np.float32)
        if config.get("oraclerep", False) else None
    )
    agent = agent_cls.create(seed, ex_obs, ex_act, config, ex_goals=ex_goals)
    agent = restore_agent(agent, str(save_dir), restore_step)
    return agent, config


# ---------------------------------------------------------------------------
# Spot driver
# ---------------------------------------------------------------------------


class _DryRunSpot:
    """Simulated Spot: integrates commanded waypoints in software."""

    def __init__(self, init_xy: np.ndarray):
        self._xy = init_xy.astype(np.float32).copy()

    def get_xy(self) -> np.ndarray:
        return self._xy.copy()

    def go_to(self, target_xy: np.ndarray, blocking: bool = True):
        self._xy = target_xy.astype(np.float32)

    def stand(self): pass
    def sit(self): pass
    def shutdown(self): pass


class _RealSpot:
    """Minimal Spot wrapper: stand → trajectory_command (one waypoint at a time)."""

    def __init__(self, hostname: str, username: str, password: str,
                 frame_offset: np.ndarray, end_time_s: float = 2.0):
        # Lazy imports so the file is importable without bosdyn installed.
        import bosdyn.client
        import bosdyn.client.lease
        from bosdyn.client.robot_command import (
            RobotCommandBuilder, RobotCommandClient, blocking_stand,
        )
        from bosdyn.client.robot_state import RobotStateClient
        from bosdyn.client.frame_helpers import (
            ODOM_FRAME_NAME, get_odom_tform_body,
        )

        self._frame_offset = np.asarray(frame_offset, dtype=np.float64)
        self._end_time_s = end_time_s
        self._RobotCommandBuilder = RobotCommandBuilder
        self._ODOM = ODOM_FRAME_NAME
        self._get_odom_tform_body = get_odom_tform_body

        sdk = bosdyn.client.create_standard_sdk("DualGoalSpotEval")
        self._robot = sdk.create_robot(hostname)
        self._robot.authenticate(username, password)
        self._robot.time_sync.wait_for_sync()
        self._lease_client = self._robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name
        )
        self._lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(
            self._lease_client, must_acquire=True, return_at_exit=True,
        )
        self._cmd_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
        self._state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
        # Power on + stand (operator must have already cleared the estop).
        self._robot.power_on(timeout_sec=20)
        blocking_stand(self._cmd_client, timeout_sec=10)

        # Anchor: the body pose at startup defines world (0, 0). Subsequent
        # get_xy() readings are subtracted from this so they live in the same
        # frame as maze.json.
        self._world_anchor = self._read_odom_body_xy()
        print(f"[spot] anchored world origin at odom xy={self._world_anchor}")

    def _read_odom_body_xy(self) -> np.ndarray:
        state = self._state_client.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        odom_T_body = self._get_odom_tform_body(snapshot)
        return np.array([odom_T_body.x, odom_T_body.y], dtype=np.float64)

    def get_xy(self) -> np.ndarray:
        body = self._read_odom_body_xy()
        return (body - self._world_anchor + self._frame_offset).astype(np.float32)

    def go_to(self, target_xy: np.ndarray, blocking: bool = True):
        # Convert world target → odom frame target.
        odom_target = (np.asarray(target_xy, dtype=np.float64)
                       + self._world_anchor - self._frame_offset)
        cmd = self._RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=float(odom_target[0]),
            goal_y=float(odom_target[1]),
            goal_heading=0.0,
            frame_name=self._ODOM,
        )
        end_time = time.time() + self._end_time_s
        cmd_id = self._cmd_client.robot_command(cmd, end_time_secs=end_time)
        if blocking:
            # Poll for arrival or timeout.
            t0 = time.time()
            while time.time() - t0 < self._end_time_s:
                cur = self.get_xy()
                if float(np.linalg.norm(cur - target_xy)) < 0.1:
                    return
                time.sleep(0.05)

    def stand(self):
        from bosdyn.client.robot_command import blocking_stand
        blocking_stand(self._cmd_client, timeout_sec=10)

    def sit(self):
        cmd = self._RobotCommandBuilder.synchro_sit_command()
        self._cmd_client.robot_command(cmd)

    def shutdown(self):
        try:
            self.sit()
            time.sleep(2.0)
        finally:
            self._robot.power_off(cut_immediately=False, timeout_sec=20)
            self._lease_keepalive.shutdown()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def _load_goal(args, maze: MazeSpec) -> np.ndarray:
    if args.goal_xy is not None:
        return np.asarray(json.loads(args.goal_xy), dtype=np.float32)
    if args.goal_cell is not None:
        rc = json.loads(args.goal_cell)
        return maze.cell_center_xy(tuple(int(v) for v in rc))
    return maze.cell_center_xy(tuple(maze.goal_cell))


def _load_start(args, maze: MazeSpec) -> np.ndarray:
    if args.start_xy is not None:
        return np.asarray(json.loads(args.start_xy), dtype=np.float32)
    if args.start_cell is not None:
        rc = json.loads(args.start_cell)
        return maze.cell_center_xy(tuple(int(v) for v in rc))
    return maze.cell_center_xy(tuple(maze.start_cell))


def parse_args():
    p = argparse.ArgumentParser(description="Eval pointmaze GCIVL-Dual on Boston Dynamics Spot.")
    # Snapshot / dataset
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--restore-step", type=int, required=True)
    p.add_argument("--maze-json", type=str, default=None,
                   help="Path to maze.json. Defaults to <save_dir>/maze.json.")

    # Goal / start (any one option)
    p.add_argument("--goal-xy", type=str, default=None, help='JSON "[x, y]" in metres.')
    p.add_argument("--goal-cell", type=str, default=None, help='JSON "[row, col]".')
    p.add_argument("--start-xy", type=str, default=None)
    p.add_argument("--start-cell", type=str, default=None)

    # Spot connection
    p.add_argument("--hostname", type=str, default=None)
    p.add_argument("--username", type=str, default=None)
    p.add_argument("--password", type=str, default=None)
    p.add_argument("--frame-offset", type=str, default="[0.0, 0.0]",
                   help='World offset to add to Spot\'s anchored xy. JSON "[dx, dy]".')

    # Episode
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--control-period-s", type=float, default=1.0,
                   help="Seconds per agent step. Spot smoothly interpolates.")
    p.add_argument("--success-tol-m", type=float, default=None,
                   help="Defaults to cell_size / 2.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    # Safety / dry-run
    p.add_argument("--dry-run", action="store_true",
                   help="No Spot connection — simulate by integrating waypoints.")
    p.add_argument("--results-path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)

    maze_path = Path(args.maze_json) if args.maze_json else save_dir / "maze.json"
    if not maze_path.exists():
        raise FileNotFoundError(
            f"{maze_path} not found. Pass --maze-json or copy maze.json into <save_dir>."
        )
    maze = MazeSpec.from_json(maze_path)

    print(f"[eval] restoring agent from {save_dir} step {args.restore_step}")
    agent, config = _build_and_restore(save_dir, args.restore_step, args.seed)

    start_xy = _load_start(args, maze)
    goal_xy = _load_goal(args, maze)
    success_tol = args.success_tol_m if args.success_tol_m is not None else 0.5 * maze.cell_size_m
    print(f"[eval] start={start_xy.tolist()}  goal={goal_xy.tolist()}  tol={success_tol:.2f} m")

    # Robot connect
    if args.dry_run:
        print("[eval] DRY RUN — using simulated Spot")
        spot = _DryRunSpot(start_xy)
    else:
        if not all([args.hostname, args.username, args.password]):
            raise ValueError("Real-robot eval requires --hostname --username --password.")
        spot = _RealSpot(
            hostname=args.hostname,
            username=args.username,
            password=args.password,
            frame_offset=np.asarray(json.loads(args.frame_offset), dtype=np.float64),
            end_time_s=max(args.control_period_s + 0.5, 1.5),
        )
        # Move Spot to the requested start cell before kicking off the policy
        # (operator should still keep the e-stop ready).
        print(f"[spot] driving to start {start_xy.tolist()}")
        spot.go_to(start_xy, blocking=True)
        time.sleep(1.0)

    @jax.jit
    def _act(obs, goal, key):
        return agent.sample_actions(
            observations=obs[None], goals=goal[None],
            seed=key, temperature=args.temperature,
        )[0]

    rng = jax.random.PRNGKey(args.seed)
    dt = float(args.control_period_s)
    results = []

    try:
        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
            episode_step = 0
            min_dist = np.inf
            success = False
            xy = spot.get_xy()
            for step in range(args.max_steps):
                t0 = time.time()
                xy = spot.get_xy()
                d = float(np.linalg.norm(xy - goal_xy))
                min_dist = min(min_dist, d)
                if d <= success_tol:
                    print(f"  ✓ reached goal at step {step}, ||Δ||={d:.3f} m")
                    success = True
                    break
                rng, key = jax.random.split(rng)
                a = np.asarray(_act(jnp.asarray(xy, dtype=jnp.float32),
                                    jnp.asarray(goal_xy, dtype=jnp.float32), key))
                a = np.clip(a, -1.0, 1.0)
                # Convert to a target waypoint in world frame.
                target = xy + STEP_SCALE * a
                # Stop the policy from driving into a wall — clamp to current
                # position if the target lands inside a wall cell.
                if not maze.is_free(maze.xy_to_cell(target)):
                    target = xy
                if args.dry_run:
                    print(f"    step {step:3d}  d={d:.3f}  a={a.tolist()}  → {target.tolist()}")
                spot.go_to(target, blocking=False)
                episode_step = step + 1
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
            results.append({
                "episode": ep + 1,
                "steps": int(episode_step),
                "success": bool(success),
                "min_dist_m": float(min_dist),
            })
    finally:
        if not args.dry_run:
            spot.shutdown()

    n_succ = sum(r["success"] for r in results)
    print("\n=== Summary ===")
    print(f"  episodes:     {args.num_episodes}")
    print(f"  successes:    {n_succ}")
    print(f"  success rate: {100.0 * n_succ / max(1, args.num_episodes):.1f}%")
    print(f"  mean min dist: {np.mean([r['min_dist_m'] for r in results]):.3f} m")

    out_path = Path(args.results_path) if args.results_path else save_dir / "eval_spot_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "save_dir": str(save_dir),
            "restore_step": args.restore_step,
            "start": start_xy.tolist(),
            "goal": goal_xy.tolist(),
            "results": results,
            "success_rate": n_succ / max(1, args.num_episodes),
        }, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
