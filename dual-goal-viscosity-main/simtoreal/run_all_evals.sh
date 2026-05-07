#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROBOT_IP=192.168.131.41
COMMON=(--robot-ip "$ROBOT_IP" --num-episodes 1 --episode-length 200 --control-hz 10 --goal-demo-idx 0)

run() {
    echo
    echo "=============================================="
    echo "[$1]"
    echo "=============================================="
    shift
    python simtoreal/eval_real.py "$@"
    read -r -p "Press enter to continue to the next eval (Ctrl-C to abort)..."
}

run "shelf_4cam FK" \
    --save-dir ./runs/shelf_4cam_fk/ --restore-step 200000 \
    --goal-from-demo ./datasets/shelf_4cam_real/shelf_4cam_real-val.npz \
    --home-npy simtoreal/image_45.npy "${COMMON[@]}"

run "shelf_4cam no-FK" \
    --save-dir ./runs/shelf_4cam_nofk/ --restore-step 25000 \
    --goal-from-demo ./datasets/shelf_4cam_real/shelf_4cam_real-val.npz \
    --home-npy simtoreal/image_45.npy "${COMMON[@]}"

run "pick_drop FK" \
    --save-dir ./runs/pick_drop_fk/ --restore-step 200000 \
    --goal-from-demo ./datasets/pick_drop_real/pick_drop_real-val.npz \
    --home-npy simtoreal/pickuppose.npy "${COMMON[@]}"

run "pick_drop no-FK" \
    --save-dir ./runs/pick_drop_nofk/ --restore-step 25000 \
    --goal-from-demo ./datasets/pick_drop_real/pick_drop_real-val.npz \
    --home-npy simtoreal/pickuppose.npy "${COMMON[@]}"

echo
echo "All 4 evals complete."
