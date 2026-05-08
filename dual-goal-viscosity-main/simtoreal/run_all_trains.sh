#!/usr/bin/env bash
# run_all_trains.sh — Train task 1 (shelf_4cam) and/or task 2 (pick_drop),
# each in FK and no-FK variants. Hyperparameters come from simtoreal/train.txt.
#
# Usage:
#   bash simtoreal/run_all_trains.sh             # both tasks
#   bash simtoreal/run_all_trains.sh --task 1    # only shelf_4cam (FK + no-FK)
#   bash simtoreal/run_all_trains.sh --task 2    # only pick_drop (FK + no-FK)

set -euo pipefail
cd "$(dirname "$0")/.."

TASK="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --task=*) TASK="${1#*=}"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$TASK" != "all" && "$TASK" != "1" && "$TASK" != "2" ]]; then
    echo "--task must be 1, 2, or omitted (defaults to both)" >&2
    exit 1
fi

train_task1() {
    local fk=$1 save_dir=$2
    echo
    echo "============================================================"
    echo "[task 1: shelf_4cam, FK=${fk}] -> ${save_dir}"
    echo "============================================================"
    python -m simtoreal.train_real \
        --dataset_dir ./datasets/shelf_4cam_real \
        --save_dir    "${save_dir}" \
        --train_steps 200000 \
        --agent.discount=0.99 \
        --agent.expectile=0.85 \
        --agent.alpha=10.0 \
        --agent.batch_size=512 \
        --agent.viscous_scale=0.001 \
        --agent.num_walks=10 \
        --agent.goalrep_dim=128 \
        --agent.value_p_curgoal=0.2 \
        --agent.value_p_trajgoal=0.5 \
        --agent.value_p_randomgoal=0.3 \
        --agent.enable_fk_regularization="${fk}"
}

train_task2() {
    local fk=$1 save_dir=$2
    echo
    echo "============================================================"
    echo "[task 2: pick_drop, FK=${fk}] -> ${save_dir}"
    echo "============================================================"
    python -m simtoreal.train_real \
        --dataset_dir ./datasets/pick_drop_real \
        --save_dir    "${save_dir}" \
        --train_steps 200000 \
        --agent.discount=0.995 \
        --agent.expectile=0.9 \
        --agent.alpha=20.0 \
        --agent.batch_size=512 \
        --agent.viscous_scale=0.002 \
        --agent.num_walks=16 \
        --agent.goalrep_dim=256 \
        --agent.value_p_curgoal=0.1 \
        --agent.value_p_trajgoal=0.6 \
        --agent.value_p_randomgoal=0.3 \
        --agent.enable_fk_regularization="${fk}"
}

if [[ "$TASK" == "all" || "$TASK" == "1" ]]; then
    train_task1 True  ./runs/shelf_4cam_fk
    train_task1 False ./runs/shelf_4cam_nofk
fi

if [[ "$TASK" == "all" || "$TASK" == "2" ]]; then
    train_task2 True  ./runs/pick_drop_fk
    train_task2 False ./runs/pick_drop_nofk
fi

echo
echo "All requested training runs complete."
