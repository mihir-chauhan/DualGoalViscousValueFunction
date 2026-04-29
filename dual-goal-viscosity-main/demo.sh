#!/bin/bash

# python demo.py \
#     --env_name=antmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py

export MUJOCO_GL=egl
# Xvfb :100 -ac &
# PID1=$!
# export DISPLAY=:100.0

# ----- select method -----
# eik
# SESS=eik_gcivl_dual_scene-play-v0_sd000_s_5621177.0.20251226_132119
# proposal
SESS=scene-play-v0_sd000_s_5337122.0.20251216_012830

python demo_mj.py \
    --env_name=scene-play-v0 \
    --agent=agents/gcivl/state/dual.py \
    --restore_path=exp/for-plots/${SESS} \
    --restore_epoch 1000000
