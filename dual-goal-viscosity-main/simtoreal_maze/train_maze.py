"""
train_maze.py — Offline GCIVL-Dual training on the synthetic real-room maze
demos produced by `simtoreal_maze.generate_demos`.

This is a thin alias of `simtoreal/train_real.py`: same agent
(`agents/gcivl/state/dual.py`), same loss, same FK-on/off comparison flow.
The only thing that's different is the dataset — 2-D pointmaze observations
and actions instead of 8-D Franka joints.

We keep the Franka + maze trainers as separate files so each stays a single
self-contained script you can `python -m` directly. If you ever change a
shared code path, edit it in `simtoreal/train_real.py` and copy here — the
two are intentionally near-identical.

Usage
-----
    # FK regularizer ON (proposed)
    python -m simtoreal_maze.train_maze \
        --dataset_dir ./datasets/maze_lab \
        --save_dir    ./runs/maze_lab_fk \
        --train_steps 500000 \
        --agent.enable_fk_regularization=True

    # Baseline (FK OFF)
    python -m simtoreal_maze.train_maze \
        --dataset_dir ./datasets/maze_lab \
        --save_dir    ./runs/maze_lab_nofk \
        --train_steps 500000 \
        --agent.enable_fk_regularization=False
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_flag_dict


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("dataset_dir", None, "Directory containing maze_real{,-val}.npz, action_stats.npz, maze.json.")
flags.DEFINE_string("dataset_name", "maze_real", "Dataset basename.")
flags.DEFINE_string("save_dir", "./runs/maze_real", "Where to save snapshots & logs.")

flags.DEFINE_integer("train_steps", 500000, "Number of update steps.")
flags.DEFINE_integer("log_interval", 1000, "Steps between train-loss logs.")
flags.DEFINE_integer("val_interval", 5000, "Steps between validation-loss logs.")
flags.DEFINE_integer("save_interval", 50000, "Steps between snapshot saves.")

flags.mark_flag_as_required("dataset_dir")

config_flags.DEFINE_config_file(
    "agent",
    str(_PROJECT_ROOT / "agents" / "gcivl" / "state" / "dual.py"),
    lock_config=False,
)


def _load_dataset(path: Path) -> dict:
    with np.load(str(path)) as f:
        out = {k: np.asarray(f[k]) for k in f.files}
    out["observations"] = out["observations"].astype(np.float32)
    out["actions"] = out["actions"].astype(np.float32)
    out["terminals"] = out["terminals"].astype(np.float32)
    if "valids" in out:
        out["valids"] = out["valids"].astype(np.float32)
    keep = {"observations", "actions", "terminals", "valids"}
    return {k: v for k, v in out.items() if k in keep}


def main(_):
    config = FLAGS.agent
    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "flags.json", "w") as f:
        json.dump(get_flag_dict(), f, indent=2)

    # -------- 1. Load dataset -----------------------------------------
    dataset_dir = Path(FLAGS.dataset_dir)
    train_path = dataset_dir / f"{FLAGS.dataset_name}.npz"
    val_path = dataset_dir / f"{FLAGS.dataset_name}-val.npz"
    stats_path = dataset_dir / "action_stats.npz"
    maze_path = dataset_dir / "maze.json"
    assert train_path.exists(), f"missing {train_path}"
    assert stats_path.exists(), f"missing {stats_path}"

    print(f"[data] loading {train_path}")
    train_raw = _load_dataset(train_path)
    val_raw = _load_dataset(val_path) if val_path.exists() else None

    with np.load(str(stats_path)) as f:
        action_stats = {"min": f["min"].astype(np.float32),
                        "max": f["max"].astype(np.float32)}
    np.savez(save_dir / "action_stats.npz", **action_stats)
    if maze_path.exists():
        # Stash the room spec alongside the snapshot so eval_spot.py can find
        # it without --maze-json.
        import shutil
        shutil.copy2(maze_path, save_dir / "maze.json")

    # Speed field for the FK regularizer (see main.py / state/dual.py).
    if "speed_profile" in config:
        if config["speed_profile"] != "constant":
            raise ValueError(
                "Real-room maze has no obstacle-distance map. Use the "
                "default --agent.speed_profile=constant."
            )
        train_raw["speed"] = np.ones((train_raw["observations"].shape[0],), dtype=np.float32)
        if val_raw is not None:
            val_raw["speed"] = np.ones((val_raw["observations"].shape[0],), dtype=np.float32)

    dataset_cls = {"GCDataset": GCDataset, "HGCDataset": HGCDataset, "VIPDataset": VIPDataset}[
        config["dataset_class"]
    ]
    train_dataset = dataset_cls(Dataset.create(norm=config["norm"], **train_raw), config)
    val_dataset = (
        dataset_cls(Dataset.create(norm=config["norm"], **val_raw), config)
        if val_raw is not None else None
    )

    # -------- 2. Build agent ------------------------------------------
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    example_batch = train_dataset.sample(1)
    if config["discrete"]:
        example_batch["actions"] = np.full_like(
            example_batch["actions"], example_batch["actions"].max() + 1
        )
    agent_class = agents[config["agent_name"]]
    ex_goals = example_batch["value_goals"] if config["oraclerep"] else None
    agent = agent_class.create(
        FLAGS.seed, example_batch["observations"], example_batch["actions"],
        config, ex_goals=ex_goals,
    )

    fk_tag = "fk" if config.get("enable_fk_regularization", True) else "nofk"
    print(f"[agent] {config['agent_name']} — regularization: {fk_tag}")
    print(f"[agent] obs_dim={example_batch['observations'].shape[-1]}  "
          f"act_dim={example_batch['actions'].shape[-1]}")

    # -------- 3. Train loop -------------------------------------------
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
            val_logger.log(
                {f"validation/{k}": float(np.asarray(v)) for k, v in val_info.items()},
                step=step,
            )

        if step % FLAGS.save_interval == 0 or step == FLAGS.train_steps:
            save_agent(agent, str(save_dir), step)

    train_logger.close()
    if val_logger is not None:
        val_logger.close()
    print(f"\nTraining done. Snapshots in {save_dir}.")


if __name__ == "__main__":
    app.run(main)
