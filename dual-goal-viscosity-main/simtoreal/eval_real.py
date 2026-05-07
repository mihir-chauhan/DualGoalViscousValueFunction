"""
eval_real.py — Deploy a trained GCIVL-Dual snapshot on a real Franka Panda.

Loads:
  - <save_dir>/params_<step>.pkl       Flax agent snapshot
  - <save_dir>/flags.json              Run config (agent + dataset name)
  - <save_dir>/action_stats.npz        Saved during train_real.py

Goals can come from one of:
  --goal-joints '[q0,...,q6,grip]'     Explicit 8-dim target state.
  --goal-from-demo path/to/dataset.npz Use the final state of a recorded
                                       trajectory in an OGBench-style npz
                                       (e.g. franka_real-val.npz).
  --goal-demo-idx N                    Which trajectory in that file to use
                                       (default: 0).

Usage
-----
    python -m simtoreal.eval_real \
        --robot-ip 192.168.131.41 \
        --save-dir ./runs/franka_real_fk \
        --restore-step 200000 \
        --goal-from-demo ./datasets/franka_real/franka_real-val.npz \
        --goal-demo-idx 0 \
        --num-episodes 5

    # Headless dry-run (no robot, just prints actions):
    python -m simtoreal.eval_real --dry-run \
        --save-dir ./runs/franka_real_fk \
        --restore-step 200000 \
        --goal-joints '[0.0,-0.4,0.0,-2.0,0.0,1.6,0.7,1.0]' \
        --num-episodes 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents import agents
from utils.flax_utils import restore_agent
from ml_collections import ConfigDict


JOINT_DIM = 7
OBS_DIM = JOINT_DIM + 1
ACT_DIM = JOINT_DIM + 1


# ---------------------------------------------------------------------------
# Config / agent reconstruction
# ---------------------------------------------------------------------------


def _load_flags_config(save_dir: Path) -> ConfigDict:
    """Reconstruct the agent ConfigDict from the flags.json saved at train time."""
    flags_path = save_dir / "flags.json"
    if not flags_path.exists():
        raise FileNotFoundError(
            f"{flags_path} not found — re-run train_real.py to regenerate."
        )
    with open(flags_path) as f:
        run_flags = json.load(f)
    # Pull out everything under the top-level "agent" key the absl-json dump
    # produces. ConfigDict handles arbitrary dicts.
    if "agent" in run_flags:
        agent_cfg = run_flags["agent"]
    else:
        # Fallback: every key starting with "agent." goes in.
        agent_cfg = {
            k.split(".", 1)[1]: v
            for k, v in run_flags.items() if k.startswith("agent.")
        }
    return ConfigDict(agent_cfg)


def _build_and_restore_agent(save_dir: Path, restore_step: int, seed: int):
    config = _load_flags_config(save_dir)
    agent_class = agents[config["agent_name"]]

    ex_obs = np.zeros((1, OBS_DIM), dtype=np.float32)
    ex_act = np.zeros((1, ACT_DIM), dtype=np.float32)
    ex_goals = (
        np.zeros((1, config["goalrep_dim"]), dtype=np.float32)
        if config.get("oraclerep", False)
        else None
    )
    agent = agent_class.create(seed, ex_obs, ex_act, config, ex_goals=ex_goals)
    agent = restore_agent(agent, str(save_dir), restore_step)
    return agent, config


# ---------------------------------------------------------------------------
# Action helpers (de-normalisation mirrors CQN-AS simtoreal/real_env.py)
# ---------------------------------------------------------------------------


def _denormalise_action(a_norm: np.ndarray, action_stats: dict) -> np.ndarray:
    a_min = action_stats["min"]
    a_max = action_stats["max"]
    a01 = (a_norm + 1.0) * 0.5
    return a_min + a01 * (a_max - a_min)


def _load_goal(args) -> np.ndarray:
    if args.goal_joints is not None:
        goal = np.asarray(json.loads(args.goal_joints), dtype=np.float32)
        if goal.shape != (OBS_DIM,):
            raise ValueError(
                f"--goal-joints must have {OBS_DIM} entries (7 joints + 1 gripper); "
                f"got shape {goal.shape}"
            )
        return goal
    if args.goal_from_demo is None:
        raise ValueError("Pass either --goal-joints or --goal-from-demo.")
    return _demo_state(args.goal_from_demo, args.goal_demo_idx, "end")


def _demo_state(npz_path: str, demo_idx: int, which: str) -> np.ndarray:
    """Return the first ('start') or last ('end') state of trajectory `demo_idx`."""
    with np.load(npz_path) as f:
        obs = np.asarray(f["observations"], dtype=np.float32)
        terms = np.asarray(f["terminals"], dtype=np.float32)
    term_locs = np.flatnonzero(terms > 0)
    # Compact OGBench datasets put terminal=1 on both the last *and* the
    # second-to-last step of every demo. Collapse adjacent runs to recover
    # one terminal index per trajectory.
    traj_ends = []
    for t in term_locs:
        if traj_ends and t == traj_ends[-1] + 1:
            traj_ends[-1] = t
        else:
            traj_ends.append(int(t))
    if demo_idx >= len(traj_ends):
        raise IndexError(
            f"demo idx {demo_idx} out of range (found {len(traj_ends)} trajectories) in {npz_path}."
        )
    if which == "end":
        return obs[traj_ends[demo_idx]].astype(np.float32)
    # First step of trajectory `demo_idx`: 0 for the first traj, else one past the previous end.
    start = 0 if demo_idx == 0 else traj_ends[demo_idx - 1] + 1
    return obs[start].astype(np.float32)


# ---------------------------------------------------------------------------
# Robot wrapper (minimal — just the bits we need for state-only rollouts)
# ---------------------------------------------------------------------------


class _DryRunRobot:
    """Stand-in for franky.Robot: simulates state by integrating commanded deltas."""

    def __init__(self, init_q: np.ndarray):
        self._q = init_q.astype(np.float32).copy()
        self._gripper_open = True

    @property
    def current_joint_state(self):
        class _S:
            position = self._q.tolist()
        s = _S()
        s.position = self._q.tolist()
        return s

    def move_to(self, target_q: np.ndarray):
        self._q = target_q.astype(np.float32)

    def set_gripper(self, open_: bool):
        self._gripper_open = bool(open_)

    @property
    def gripper_open(self) -> bool:
        return self._gripper_open


class _RealRobot:
    """Thin franky wrapper. Imports happen lazily so dry-run doesn't need it."""

    def __init__(self, robot_ip: str, joint_delta_clip: float, velocity_factor: float,
                 gripper_force: float):
        from franky import (ControlException, Gripper, JointWaypoint,
                JointWaypointMotion, ReferenceType,
                RelativeDynamicsFactor, Robot)
        self._JointWaypoint = JointWaypoint
        self._JointWaypointMotion = JointWaypointMotion
        self._ReferenceType = ReferenceType
        self._RelDyn = RelativeDynamicsFactor
        self._ControlException = ControlException
        self._delta_clip = float(joint_delta_clip)
        self._gripper_force = float(gripper_force)
        self._velocity_factor = float(velocity_factor)

        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)
        self.robot.set_collision_behavior(50, 50)
        self.robot.recover_from_errors()
        # Per-call cap; we additionally use a much *softer* dynamics factor
        # for the tiny streamed waypoints so consecutive sends don't generate
        # velocity / acceleration discontinuities when overridden mid-motion.
        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(
            velocity_factor, velocity_factor, velocity_factor,
        )
        self._stream_dyn = RelativeDynamicsFactor(
            min(velocity_factor, 0.05),
            min(velocity_factor, 0.05),
            min(velocity_factor, 0.05),
        )
        self.gripper.open(0.1)
        self._gripper_is_open = True
        self._homed_once = False

    def _recover_from_motion_error(self):
        try:
            self.robot.recover_from_errors()
        except Exception:
            pass

    @property
    def current_joint_state(self):
        return self.robot.current_joint_state

    def move_home(self, target_q: np.ndarray):
        """Synchronous, full-speed move used at episode start. Comes to a stop."""
        target = np.asarray(target_q, dtype=np.float64).tolist()
        motion = self._JointWaypointMotion([
            self._JointWaypoint(target, reference_type=self._ReferenceType.Absolute)
        ])
        try:
            self.robot.move(motion, asynchronous=False)
            self._homed_once = True
            return True
        except self._ControlException as exc:
            print(f"[eval] warning: home motion failed, recovering and continuing: {exc}")
            self._recover_from_motion_error()
            return False

    def move_to(self, target_q: np.ndarray):
        """Streaming step. Asynchronous so the controller doesn't decelerate to
        zero between waypoints; new sends override the prior target smoothly."""
        target = np.asarray(target_q, dtype=np.float64).tolist()
        wp = self._JointWaypoint(
            target, reference_type=self._ReferenceType.Absolute
        )
        motion = self._JointWaypointMotion(
            [wp], relative_dynamics_factor=self._stream_dyn,
        )
        try:
            self.robot.move(motion, asynchronous=True)
        except TypeError:
            # Older franky versions don't accept asynchronous kwarg — fall back.
            try:
                self.robot.move(motion)
            except self._ControlException as exc:
                print(f"[eval] warning: streaming motion failed, recovering and continuing: {exc}")
                self._recover_from_motion_error()
                return False
        except self._ControlException as exc:
            print(f"[eval] warning: streaming motion failed, recovering and continuing: {exc}")
            self._recover_from_motion_error()
            return False
        return True

    def set_gripper(self, open_: bool):
        if open_ == self._gripper_is_open:
            return
        if open_:
            self.gripper.open(0.1)
        else:
            self.gripper.grasp(0.0, 0.1, self._gripper_force,
                               epsilon_inner=1.0, epsilon_outer=1.0)
        self._gripper_is_open = open_

    @property
    def gripper_open(self) -> bool:
        return self._gripper_is_open


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Eval a trained GCIVL-Dual snapshot on real Franka.")
    p.add_argument("--robot-ip", type=str, default="192.168.131.41")
    p.add_argument("--save-dir", type=str, required=True,
                   help="Directory containing params_<step>.pkl, flags.json, action_stats.npz")
    p.add_argument("--restore-step", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)

    # Goal selection
    p.add_argument("--goal-joints", type=str, default=None,
                   help='JSON list of 8 floats: [q0..q6, gripper_open]')
    p.add_argument("--goal-from-demo", type=str, default=None,
                   help="Path to an OGBench-style npz to pull a goal from.")
    p.add_argument("--goal-demo-idx", type=int, default=0)

    # Episode
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--episode-length", type=int, default=400)
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--success-joint-thresh", type=float, default=0.05,
                   help="Success when ||q - q_goal||_2 (radians) <= this value.")
    p.add_argument("--joint-delta-clip", type=float, default=0.05)
    p.add_argument("--velocity-factor", type=float, default=0.15)
    p.add_argument("--gripper-force", type=float, default=20.0)
    p.add_argument("--temperature", type=float, default=0.0)

    # Home pose (.npy 7-dof) — same convention as the CQN-AS simtoreal scripts.
    p.add_argument("--home-q", type=str, default=None,
                   help="JSON list of 7 joint angles for home pose.")
    p.add_argument("--home-npy", type=str, nargs="+", default=None,
                   help="One or more .npy files (each 7 joint angles). If multiple, "
                        "they are visited in order as a homing sequence.")
    p.add_argument("--skip-home", action="store_true")

    # Dry-run / safety
    p.add_argument("--dry-run", action="store_true",
                   help="Don't talk to the robot. Simulate by integrating deltas.")
    p.add_argument("--results-path", type=str, default=None,
                   help="JSON file to save per-episode results (default: <save_dir>/eval_results.json)")
    return p.parse_args()


def _read_obs(robot) -> np.ndarray:
    q = np.asarray(robot.current_joint_state.position, dtype=np.float32)[:JOINT_DIM]
    g = 1.0 if robot.gripper_open else 0.0
    return np.concatenate([q, np.array([g], dtype=np.float32)])


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    stats_path = save_dir / "action_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found — train_real.py copies action_stats next "
            "to its snapshots; check --save-dir."
        )
    with np.load(str(stats_path)) as f:
        action_stats = {"min": f["min"].astype(np.float32),
                        "max": f["max"].astype(np.float32)}

    print(f"[eval] restoring agent from {save_dir} step {args.restore_step}")
    agent, config = _build_and_restore_agent(save_dir, args.restore_step, args.seed)

    goal = _load_goal(args)
    print(f"[eval] goal (8-dim): {np.array2string(goal, precision=3)}")

    if args.home_npy:
        home_qs = []
        for path in args.home_npy:
            wp = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
            if wp.shape != (JOINT_DIM,):
                raise ValueError(
                    f"{path}: expected shape ({JOINT_DIM},), got {wp.shape}"
                )
            home_qs.append(wp)
    elif args.home_q is not None:
        home_qs = [np.asarray(json.loads(args.home_q), dtype=np.float32)]
    else:
        home_qs = [np.asarray(goal[:JOINT_DIM], dtype=np.float32)]

    # Robot connection (lazy: dry-run avoids importing franky entirely).
    if args.dry_run:
        print("[eval] DRY RUN — using simulated robot")
        robot = _DryRunRobot(home_qs[-1])
    else:
        print(f"[eval] connecting to robot at {args.robot_ip}")
        robot = _RealRobot(
            robot_ip=args.robot_ip,
            joint_delta_clip=args.joint_delta_clip,
            velocity_factor=args.velocity_factor,
            gripper_force=args.gripper_force,
        )

    # Pre-compile a sample_actions wrapper that handles the ([1, D]) batching.
    @jax.jit
    def _act(obs, goal, key):
        return agent.sample_actions(
            observations=obs[None], goals=goal[None],
            seed=key, temperature=args.temperature,
        )[0]

    rng = jax.random.PRNGKey(args.seed)
    dt = 1.0 / args.control_hz
    results = []

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        if not args.skip_home:
            home_fn = getattr(robot, "move_home", robot.move_to)
            home_ok = True
            for i, wp in enumerate(home_qs):
                tag = f" {i+1}/{len(home_qs)}" if len(home_qs) > 1 else ""
                print(f"  [home{tag}] moving to {np.array2string(wp, precision=3)}")
                ok = home_fn(wp)
                if ok is False:
                    home_ok = False
            robot.set_gripper(True)
            time.sleep(1.0)
            # Make sure the controller is fully settled before we start
            # streaming small waypoints — otherwise the first streamed
            # send still sees nonzero velocity from the home motion and
            # the reflex fires.
            try:
                robot.robot.recover_from_errors()  # type: ignore[attr-defined]
            except AttributeError:
                pass
            if not home_ok:
                print("  [home] continuing after motion error")

        episode_step = 0
        success = False
        episode_min_dist = np.inf
        for step in range(args.episode_length):
            t0 = time.time()
            obs = _read_obs(robot)
            dist = float(np.linalg.norm(obs[:JOINT_DIM] - goal[:JOINT_DIM]))
            episode_min_dist = min(episode_min_dist, dist)
            if dist <= args.success_joint_thresh:
                print(f"  ✓ Success at step {step}: ||Δq||={dist:.4f}")
                # Match the demo's gripper command at the goal.
                robot.set_gripper(bool(goal[JOINT_DIM] >= 0.5))
                success = True
                break

            rng, key = jax.random.split(rng)
            a_norm = np.asarray(_act(jnp.asarray(obs), jnp.asarray(goal), key))
            a_norm = np.clip(a_norm, -1.0, 1.0)
            action = _denormalise_action(a_norm, action_stats)

            delta_q = action[:JOINT_DIM]
            # Safety clip per-step delta — same convention as the CQN-AS env.
            delta_q = np.clip(delta_q, -args.joint_delta_clip, args.joint_delta_clip)
            target_q = np.asarray(robot.current_joint_state.position, dtype=np.float32)[:JOINT_DIM] + delta_q
            grip_target = bool(action[JOINT_DIM] >= 0.5)

            if args.dry_run:
                print(f"    step {step:3d}  d={dist:.3f}  Δq={np.array2string(delta_q, precision=3)}  "
                      f"grip={'O' if grip_target else 'C'}")
            robot.move_to(target_q)
            robot.set_gripper(grip_target)

            episode_step = step + 1
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

        results.append({
            "episode": ep + 1,
            "steps": int(episode_step),
            "success": bool(success),
            "min_joint_dist": float(episode_min_dist),
        })
        print(f"  episode result: success={success}, min ||Δq||={episode_min_dist:.4f}")

    n_succ = sum(r["success"] for r in results)
    print("\n=== Summary ===")
    print(f"  episodes:     {args.num_episodes}")
    print(f"  successes:    {n_succ}")
    print(f"  success rate: {100.0 * n_succ / max(1, args.num_episodes):.1f}%")
    print(f"  mean min Δq:  {np.mean([r['min_joint_dist'] for r in results]):.4f}")

    out_path = Path(args.results_path) if args.results_path else save_dir / "eval_results.json"
    summary = {
        "save_dir": str(save_dir),
        "restore_step": args.restore_step,
        "goal": goal.tolist(),
        "num_episodes": args.num_episodes,
        "success_rate": n_succ / max(1, args.num_episodes),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
