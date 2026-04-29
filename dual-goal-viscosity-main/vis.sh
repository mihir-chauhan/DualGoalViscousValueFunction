#!/bin/bash

black vis.py

# ----- testing -----
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.alpha=10.0 \
#     --agent.rep_type=bilinear \
#     --agent.rep_expectile=0.9 \
#     --agent.goalrep_dim=64 \
#     --agent.discount=0.99 \
#     --restore_path=exp/goal_representation/Debug/pointmaze-medium-navigate-v0_sd000_20251218_194334 \
#     --restore_epoch 1000000

# python vis.py \
#     --env_name=antmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.alpha=10.0 \
#     --agent.rep_type=bilinear \
#     --agent.rep_expectile=0.9 \
#     --agent.goalrep_dim=256 \
#     --agent.discount=0.99 \
#     --restore_path=exp/goal_representation/Debug/antmaze-medium-navigate-v0_sd000_20251220_054059 \
#     --restore_epoch 1000000

#----------------------------------------------------------
# ----- for paper -----

# no sessions for antmaze-medium-navigate-v0
# python vis.py \
#     --env_name=antmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.alpha=10.0 \
#     --agent.rep_type=bilinear \
#     --agent.rep_expectile=0.9 \
#     --agent.goalrep_dim=256 \
#     --agent.discount=0.99 \
#     --restore_path=exp/Debug/antmaze-medium-navigate-v0_sd000_s_5394034.0.20251219_171820 \
#     --restore_epoch 1000000

# # need to check dof
# python vis.py \
#     --env_name=antsoccer-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.alpha=10.0 \
#     --agent.rep_type=bilinear \
#     --agent.rep_expectile=0.9 \
#     --agent.goalrep_dim=256 \
#     --agent.discount=0.99 \
#     --restore_path=exp/Debug/antsoccer-medium-navigate-v0_sd000_s_5347735.0.20251217_220815 \
#     --restore_epoch 1000000

# -------------------- antmaze-medium-navigate-v0 --------------------
# # works
# python vis.py \
#     --env_name=antmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.alpha=10.0 \
#     --agent.rep_type=bilinear \
#     --agent.rep_expectile=0.9 \
#     --agent.goalrep_dim=256 \
#     --agent.discount=0.99 \
#     --restore_path=exp/Debug/eik_gcivl_dual_antmaze-medium-navigate-v0_sd000_s_5691504.0.20251231_144815 \
#     --restore_epoch 1000000

# -------------------- pointmaze-medium-navigate-v0 --------------------
# ours
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.001_pointmaze-large-navigate-v0_sd000_s_7137846.0.20260128_220921\
#     --restore_epoch 1000000

# # eik-gcivl-dual
python vis.py \
    --env_name=pointmaze-large-navigate-v0 \
    --agent=agents/gcivl/state/dual.py \
    --agent.goalrep_dim=64 \
    --restore_path=exp/goal_representation/Debug/eik_gcivl_dual_pointmaze-large-navigate-v0_sd001_s_6854524.0.20260124_183639 \
    --restore_epoch 1000000

# # fk_original
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/original.py \
#     --restore_path=exp/goal_representation/Debug/gcivl_fk_pointmaze-large-navigate-v0_sd000_s_7104677.0.20260127_175825 \
#     --restore_epoch 1000000

# # VIB_NO_FK
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/vib.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/gcivl_vib_no_fk_pointmaze-large-navigate-v0_sd000_s_7104678.0.20260127_175825 \
#     --restore_epoch 1000000

# # orig_no_fk
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/original.py \
#     --restore_path=exp/goal_representation/Debug/gcivl_no_fk_pointmaze-large-navigate-v0_sd000_s_7104676.0.20260127_175842 \
#     --restore_epoch 1000000

# # vib with FK
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/vib.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/gcivl_vib_fk_pointmaze-large-navigate-v0_sd000_s_7104679.0.20260127_180628 \
#     --restore_epoch 1000000
# # dual no fk
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/gcivl_dual_no_fk_pointmaze-large-navigate-v0_sd000_s_7105640.0.20260127_185944 \
#     --restore_epoch 1000000

# # VIB Eikonal
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/eikonal_vib.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/gcivl_eik_vib_fk_pointmaze-large-navigate-v0_sd000_s_7105641.0.20260127_190817 \
#     --restore_epoch 1000000

# # gcivl
# python vis.py \
#     --env_name=humanoidmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=256 \
#     --restore_path=exp/goal_representation/Debug/gcivl_humanoidmaze-medium-navigate-v0_sd000_s_6148464.0.20260110_195131 \
#     --restore_epoch 1000000

# # gcvib-dual
# python vis.py \
#     --env_name=humanoidmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/vib.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/gcivl_vib_humanoidmaze-medium-navigate-v0_sd000_s_6314269.0.20260118_005113 \
#     --restore_epoch 1000000

# # gcivl-byol
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/byol.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/Debug/gcivl_byol_pointmaze-medium-navigate-v0_sd000_s_6187599.0.20260111_134752 \
#     --restore_epoch 1000000


#Ablations
# 1. Large | Viscous False | nwalks 10 | scale 0.001
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_False_nwalks_10_nuscale_0.001_pointmaze-large-navigate-v0_sd000_s_7043701.0.20260126_225618 \
#     --restore_epoch 1000000

# # 2. Large | Viscous False | nwalks 10 | scale 0.001 (Duplicate config, diff seed/run)
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_False_nwalks_10_nuscale_0.001_pointmaze-large-navigate-v0_sd000_s_7049734.0.20260126_231403 \
#     --restore_epoch 1000000

# # 3. Large | Viscous True | nwalks 10 | scale 0.0001
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.0001_pointmaze-large-navigate-v0_sd000_s_7043699.0.20260126_224955 \
#     --restore_epoch 1000000

# # 4. Medium | Viscous True | nwalks 10 | scale 0.0001
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.0001_pointmaze-medium-navigate-v0_sd000_s_7043294.0.20260126_211453 \
#     --restore_epoch 1000000

# # 5. Medium | Viscous True | nwalks 10 | scale 0.0001 (Duplicate config)
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.0001_pointmaze-medium-navigate-v0_sd000_s_7043699.0.20260126_211739 \
#     --restore_epoch 1000000

# # 6. Medium | Viscous True | nwalks 10 | scale 0.001
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.001_pointmaze-medium-navigate-v0_sd000_s_7043701.0.20260126_212007 \
#     --restore_epoch 1000000

# # 7. Large | Viscous True | nwalks 10 | scale 0.01
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.01_pointmaze-large-navigate-v0_sd000_s_7043699.0.20260127_020040 \
#     --restore_epoch 1000000

# # 8. Medium | Viscous True | nwalks 10 | scale 0.01
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.01_pointmaze-medium-navigate-v0_sd000_s_7043699.0.20260127_003557 \
#     --restore_epoch 1000000

# # 9. Large | Viscous True | nwalks 10 | scale 0.1
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.1_pointmaze-large-navigate-v0_sd000_s_7043699.0.20260127_050639 \
#     --restore_epoch 1000000

# # 10. Medium | Viscous True | nwalks 10 | scale 0.1
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_10_nuscale_0.1_pointmaze-medium-navigate-v0_sd000_s_7043699.0.20260127_034150 \
#     --restore_epoch 1000000

# # 11. Large | Viscous True | nwalks 1 | scale 0.001
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_1_nuscale_0.001_pointmaze-large-navigate-v0_sd000_s_7043700.0.20260126_221659 \
#     --restore_epoch 1000000

# # 12. Medium | Viscous True | nwalks 1 | scale 0.001
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_1_nuscale_0.001_pointmaze-medium-navigate-v0_sd000_s_7043700.0.20260126_211803 \
#     --restore_epoch 1000000

# # 13. Large | Viscous True | nwalks 20 | scale 0.001
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_20_nuscale_0.001_pointmaze-large-navigate-v0_sd000_s_7043700.0.20260127_043717 \
#     --restore_epoch 1000000

# # 14. Medium | Viscous True | nwalks 20 | scale 0.001
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_20_nuscale_0.001_pointmaze-medium-navigate-v0_sd000_s_7043700.0.20260127_021130 \
#     --restore_epoch 1000000

# # 15. Large | Viscous True | nwalks 5 | scale 0.001
# python vis.py \
#     --env_name=pointmaze-large-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_5_nuscale_0.001_pointmaze-large-navigate-v0_sd000_s_7043700.0.20260127_003549 \
#     --restore_epoch 1000000

# # 16. Medium | Viscous True | nwalks 5 | scale 0.001
# python vis.py \
#     --env_name=pointmaze-medium-navigate-v0 \
#     --agent=agents/gcivl/state/dual.py \
#     --agent.goalrep_dim=64 \
#     --restore_path=exp/goal_representation/Debug/verbose_gcivl_dual_fk_viscous_True_nwalks_5_nuscale_0.001_pointmaze-medium-navigate-v0_sd000_s_7043700.0.20260126_233031 \
#     --restore_epoch 1000000