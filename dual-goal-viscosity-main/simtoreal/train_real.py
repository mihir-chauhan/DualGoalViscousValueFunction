"""
train_real.py — Offline GCIVL-Dual training on real-Franka waypoint demos.

This is the project's `main.py`, but stripped of OGBench env construction and
online evaluation. It loads an OGBench-style npz dataset (produced by
`simtoreal/convert_waypoints.py` from CQN-AS waypoint .json files) and runs
the same agent update loop. Real-robot evaluation is delegated to
`simtoreal/eval_real.py`.

Why two scripts: the dual-goal-viscosity training pipeline assumes a
goal-conditioned simulator with `task_infos` for periodic eval. We do not have
that on the real robot, and we definitely don't want a JAX training loop
talking to a physical Franka. Train offline → save snapshots → run the
real-robot eval as a separate process.

Comparison workflow (FK regularization vs baseline):
    # proposed (FK regularizer ON)
    python -m simtoreal.train_real --dataset-dir ./datasets/franka_real \
        --save-dir ./runs/franka_real_fk --enable-fk-regularization=True
    # baseline (FK regularizer OFF)
    python -m simtoreal.train_real --dataset-dir ./datasets/franka_real \
        --save-dir ./runs/franka_real_nofk --enable-fk-regularization=False
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

# Make the dual-goal-viscosity package importable when this file is run as a
# module from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_flag_dict


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("dataset_dir", None, "Directory holding <name>.npz, <name>-val.npz, action_stats.npz.")
flags.DEFINE_string("dataset_name", "franka_real", "Dataset basename.")
flags.DEFINE_string("save_dir", "./runs/franka_real", "Where to save snapshots & logs.")

flags.DEFINE_integer("train_steps", 200000, "Number of update steps.")
flags.DEFINE_integer("log_interval", 1000, "Steps between train-loss logs.")
flags.DEFINE_integer("val_interval", 5000, "Steps between validation-loss logs.")
flags.DEFINE_integer("save_interval", 25000, "Steps between snapshot saves.")
flags.DEFINE_integer("verbose", 0, "Verbose flag (mirrors main.py).")

flags.mark_flag_as_required("dataset_dir")

# Default to the state-based GCIVL-dual config; this is the agent the proposed
# FK-regularized loss lives in (agents/gcivl/state/dual.py).
config_flags.DEFINE_config_file(
    "agent",
    str(_PROJECT_ROOT / "agents" / "gcivl" / "state" / "dual.py"),
    lock_config=False,
)


def _load_dataset(path: Path) -> dict:
    """Load an OGBench-style compact npz into a plain dict of numpy arrays."""
    with np.load(str(path)) as f:
        out = {}
        for k in f.files:
            out[k] = np.asarray(f[k])
    if "observations" not in out or "actions" not in out or "terminals" not in out:
        raise ValueError(
            f"{path} missing required keys; got {list(out.keys())}. "
            "Re-generate with simtoreal/convert_waypoints.py."
        )
    out["observations"] = out["observations"].astype(np.float32)
    out["actions"] = out["actions"].astype(np.float32)
    out["terminals"] = out["terminals"].astype(np.float32)
    if "valids" in out:
        out["valids"] = out["valids"].astype(np.float32)
    # Drop fields the GCDataset doesn't expect; keep only the canonical ones.
    keep = {"observations", "actions", "terminals", "valids"}
    return {k: v for k, v in out.items() if k in keep}


def _normalise_actions(dataset: dict, action_stats: dict) -> None:
    """Map raw action deltas → [-1, 1] in-place using saved min/max stats."""
    a = dataset["actions"]
    a_min = action_stats["min"].astype(np.float32)
    a_max = action_stats["max"].astype(np.float32)
    span = np.maximum(a_max - a_min, 1e-6)
    norm = 2.0 * (a - a_min) / span - 1.0
    dataset["actions"] = np.clip(norm, -1.0, 1.0).astype(np.float32)


def main(_):
    config = FLAGS.agent
    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Persist run config for later eval / analysis.
    flag_dict = get_flag_dict()
    with open(save_dir / "flags.json", "w") as f:
        json.dump(flag_dict, f, indent=2)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset_dir = Path(FLAGS.dataset_dir)
    train_path = dataset_dir / f"{FLAGS.dataset_name}.npz"
    val_path = dataset_dir / f"{FLAGS.dataset_name}-val.npz"
    stats_path = dataset_dir / "action_stats.npz"
    assert train_path.exists(), f"missing {train_path}"
    assert stats_path.exists(), f"missing {stats_path}"

    print(f"[data] loading {train_path}")
    train_raw = _load_dataset(train_path)
    val_raw = _load_dataset(val_path) if val_path.exists() else None

    print(f"[data] loading action stats from {stats_path}")
    with np.load(str(stats_path)) as f:
        action_stats = {"min": f["min"].astype(np.float32),
                        "max": f["max"].astype(np.float32)}
    _normalise_actions(train_raw, action_stats)
    if val_raw is not None:
        _normalise_actions(val_raw, action_stats)

    # Persist a copy of action_stats inside save_dir so eval_real.py finds it
    # alongside the snapshot regardless of where the dataset was generated.
    np.savez(save_dir / "action_stats.npz", **action_stats)

    # The GCIVL-Dual value loss expects a per-state `speed` field whenever
    # `speed_profile` is set in the agent config. Mirror main.py: speed is
    # constant on the real robot (no obstacle map) so we just fill ones.
    if "speed_profile" in config:
        if config["speed_profile"] != "constant":
            raise ValueError(
                "Real-robot data has no obstacle map. Set "
                "--agent.speed_profile=constant (the default for state/dual.py)."
            )
        train_raw["speed"] = np.ones((train_raw["observations"].shape[0],), dtype=np.float32)
        if val_raw is not None:
            val_raw["speed"] = np.ones((val_raw["observations"].shape[0],), dtype=np.float32)

    # Wrap into the project's Dataset / GCDataset just like main.py does.
    dataset_cls = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
        "VIPDataset": VIPDataset,
    }[config["dataset_class"]]
    train_dataset = dataset_cls(Dataset.create(norm=config["norm"], **train_raw), config)
    val_dataset = (
        dataset_cls(Dataset.create(norm=config["norm"], **val_raw), config)
        if val_raw is not None else None
    )

    # ------------------------------------------------------------------
    # 2. Build agent
    # ------------------------------------------------------------------
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config["discrete"]:
        # Real-robot Franka actions are continuous; this is here just to match
        # main.py's contract.
        example_batch["actions"] = np.full_like(
            example_batch["actions"], example_batch["actions"].max() + 1
        )

    agent_class = agents[config["agent_name"]]
    ex_goals = example_batch["value_goals"] if config["oraclerep"] else None
    agent = agent_class.create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
        ex_goals=ex_goals,
    )

    fk_tag = "fk" if config.get("enable_fk_regularization", True) else "nofk"
    print(f"[agent] {config['agent_name']} — regularization: {fk_tag}")
    print(f"[agent] obs_dim={example_batch['observations'].shape[-1]}  "
          f"act_dim={example_batch['actions'].shape[-1]}  "
          f"goalrep_dim={config['goalrep_dim']}")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    train_logger = CsvLogger(str(save_dir / "train.csv"))
    val_logger = CsvLogger(str(save_dir / "val.csv")) if val_dataset is not None else None
    t0 = time.time()
    last = t0

    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config["batch_size"])
        batch["global_step"] = step
        agent, info = agent.update(batch)

        if step % FLAGS.log_interval == 0:
            metrics = {f"training/{k}": float(np.asarray(v)) for k, v in info.items()}
            metrics["time/epoch_time"] = (time.time() - last) / FLAGS.log_interval
            metrics["time/total_time"] = time.time() - t0
            last = time.time()
            train_logger.log(metrics, step=step)

        if val_dataset is not None and step % FLAGS.val_interval == 0:
            val_batch = val_dataset.sample(config["batch_size"])
            val_batch["global_step"] = step
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            val_metrics = {f"validation/{k}": float(np.asarray(v)) for k, v in val_info.items()}
            val_logger.log(val_metrics, step=step)

        if step % FLAGS.save_interval == 0 or step == FLAGS.train_steps:
            save_agent(agent, str(save_dir), step)

    train_logger.close()
    if val_logger is not None:
        val_logger.close()
    print(f"\nTraining done. Snapshots in {save_dir}.")


if __name__ == "__main__":
    app.run(main)
