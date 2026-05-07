"""Sweep dry-run evals across checkpoints for all 4 (task, fk/nofk) configs."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

CONFIGS = [
    {"name": "shelf_4cam_fk",   "save_dir": "runs/shelf_4cam_fk",
     "goal":  "datasets/shelf_4cam_real/shelf_4cam_real-val.npz",
     "home":  "simtoreal/image_45.npy"},
    {"name": "shelf_4cam_nofk", "save_dir": "runs/shelf_4cam_nofk",
     "goal":  "datasets/shelf_4cam_real/shelf_4cam_real-val.npz",
     "home":  "simtoreal/image_45.npy"},
    {"name": "pick_drop_fk",    "save_dir": "runs/pick_drop_fk",
     "goal":  "datasets/pick_drop_real/pick_drop_real-val.npz",
     "home":  "simtoreal/pickuppose.npy"},
    {"name": "pick_drop_nofk",  "save_dir": "runs/pick_drop_nofk",
     "goal":  "datasets/pick_drop_real/pick_drop_real-val.npz",
     "home":  "simtoreal/pickuppose.npy"},
]
STEPS = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]

EP_LEN = 200
NUM_EP = 1


def run_one(cfg, step):
    out_json = ROOT / cfg["save_dir"] / f"_dryrun_step{step}.json"
    cmd = [
        PY, "simtoreal/eval_real.py", "--dry-run",
        "--save-dir", cfg["save_dir"], "--restore-step", str(step),
        "--goal-from-demo", cfg["goal"], "--goal-demo-idx", "0",
        "--home-npy", cfg["home"],
        "--num-episodes", str(NUM_EP),
        "--episode-length", str(EP_LEN),
        "--control-hz", "10",
        "--results-path", str(out_json),
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        return {"ok": False, "err": p.stderr.strip().splitlines()[-3:]}
    try:
        with open(out_json) as f:
            res = json.load(f)
        eps = res.get("episodes") or res.get("results") or []
        if isinstance(res, dict) and "min_joint_dist" in str(res):
            # Best-effort: harness keys might be top-level
            pass
        # eval_real.py saves a dict containing per-episode 'results' list.
        if "results" in res:
            dists = [e["min_joint_dist"] for e in res["results"]]
            succs = [e["success"] for e in res["results"]]
        else:
            dists = [e["min_joint_dist"] for e in eps]
            succs = [e["success"] for e in eps]
        return {"ok": True, "min_dist": min(dists), "succ": any(succs)}
    except Exception as e:
        return {"ok": False, "err": [f"parse err: {e}"]}


def main():
    summary = {c["name"]: {} for c in CONFIGS}
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        for step in STEPS:
            r = run_one(cfg, step)
            summary[cfg["name"]][step] = r
            tag = "OK " if r["ok"] else "ERR"
            extra = (f"min_dist={r['min_dist']:.4f}  succ={r['succ']}"
                     if r["ok"] else " | ".join(r.get("err", [])))
            print(f"  step {step:>6}  [{tag}]  {extra}")

    print("\n\n=== Best checkpoint per config (lowest min_joint_dist) ===")
    best = {}
    for name, by_step in summary.items():
        ok = {s: r["min_dist"] for s, r in by_step.items() if r["ok"]}
        if not ok:
            print(f"  {name}: NO successful runs")
            continue
        b_step = min(ok, key=ok.get)
        best[name] = (b_step, ok[b_step])
        print(f"  {name}: step {b_step}  min_dist={ok[b_step]:.4f}")

    out = ROOT / "simtoreal" / "dryrun_sweep_summary.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "best": best}, f, indent=2)
    print(f"\nFull summary written to {out}")


if __name__ == "__main__":
    main()
