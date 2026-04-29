# for type in orig_nofk orig vib_nofk vib dual eikonal; do
#   sbatch --export=ALL,JOB_TYPE=$type --job-name=pm_$type run_pointmaze_for_vis.sh
# done

for type in dual eikonal; do
  sbatch --export=ALL,JOB_TYPE=$type --job-name=pm_$type run_pointmaze_for_vis.sh
done