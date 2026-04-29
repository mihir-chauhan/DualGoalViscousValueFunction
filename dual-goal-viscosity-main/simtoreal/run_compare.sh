#!/usr/bin/env bash
# run_compare.sh — Train two GCIVL-Dual agents back-to-back on the same Franka
# real-robot dataset (FK regularizer ON vs OFF) and run eval_real.py against
# both. Copy this script and tweak paths/IPs as needed.
#
# Usage:
#   bash simtoreal/run_compare.sh \
#       /path/to/waypoints \
#       ./datasets/franka_real \
#       ./runs/franka_real \
#       <robot_ip>
#
# Args (positional):
#   1) waypoint_dir : directory of waypoints_*.json from CQN-AS record_waypoints.py
#   2) dataset_dir  : where convert_waypoints.py writes franka_real{,-val}.npz
#   3) save_root    : parent dir for the two run subfolders
#   4) robot_ip     : Franka IP (omit or "" to skip eval, train only)

set -euo pipefail
cd "$(dirname "$0")/.."

WAYPOINT_DIR=${1:?"waypoint_dir required"}
DATASET_DIR=${2:?"dataset_dir required"}
SAVE_ROOT=${3:?"save_root required"}
ROBOT_IP=${4:-""}

# 1. Build dataset (idempotent — skip if already present)
if [[ ! -f "${DATASET_DIR}/franka_real.npz" ]]; then
    echo "[1/3] Converting waypoints → OGBench-style npz"
    python -m simtoreal.convert_waypoints \
        --waypoint-dir "${WAYPOINT_DIR}" \
        --out-dir      "${DATASET_DIR}"
else
    echo "[1/3] Dataset already exists at ${DATASET_DIR}, skipping conversion"
fi

# 2. Train both variants (proposed: FK regularizer ON, baseline: OFF)
TRAIN_STEPS=${TRAIN_STEPS:-200000}

FK_DIR="${SAVE_ROOT}_fk"
NOFK_DIR="${SAVE_ROOT}_nofk"

echo "[2/3] Training FK-regularized run → ${FK_DIR}"
python -m simtoreal.train_real \
    --dataset_dir "${DATASET_DIR}" \
    --save_dir    "${FK_DIR}" \
    --train_steps "${TRAIN_STEPS}" \
    --agent.enable_fk_regularization=True

echo "[2/3] Training baseline (no FK) → ${NOFK_DIR}"
python -m simtoreal.train_real \
    --dataset_dir "${DATASET_DIR}" \
    --save_dir    "${NOFK_DIR}" \
    --train_steps "${TRAIN_STEPS}" \
    --agent.enable_fk_regularization=False

# 3. Optionally eval both on the real robot
if [[ -n "${ROBOT_IP}" ]]; then
    echo "[3/3] Evaluating both checkpoints on ${ROBOT_IP}"
    GOAL_NPZ="${DATASET_DIR}/franka_real-val.npz"
    for run in "${FK_DIR}" "${NOFK_DIR}"; do
        python -m simtoreal.eval_real \
            --robot-ip       "${ROBOT_IP}" \
            --save-dir       "${run}" \
            --restore-step   "${TRAIN_STEPS}" \
            --goal-from-demo "${GOAL_NPZ}" \
            --num-episodes   5
    done
else
    echo "[3/3] No robot IP supplied — skipping real-robot eval."
    echo "       Run later with: python -m simtoreal.eval_real --save-dir <run> --restore-step ${TRAIN_STEPS} --goal-from-demo ${DATASET_DIR}/franka_real-val.npz --robot-ip <ip>"
fi
