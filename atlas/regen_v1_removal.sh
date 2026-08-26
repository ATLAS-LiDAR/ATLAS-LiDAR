#!/usr/bin/env bash
# Regenerate the v1 (atlas_val) nuScenes REMOVAL splits with the corrected
# keyframe dt.
#
# WHY: match_with_velocity() hardcoded dt=0.1 (Waymo, 10 Hz). nuScenes annotated
# keyframes are 2 Hz, so the constant-velocity prediction was 5x too short and the
# residual for a vehicle above ~15 m/s exceeded the 6 m spatial gate -- the track
# dropped and removal silently stopped for the rest of the window. Measured on the
# original splits: median 4 of 8 attack frames actually carried a removal, and only
# 29% of scenes held all 8.
#
# FIX (already in the code): dt is now a parameter; generate_data_removal_nuscenes.py
# passes NUSC_KEYFRAME_DT=0.5 and NUSC_MAX_STEP_M=12.0 (real keyframe displacements
# reach 6.3 m, which the old 6.0 m gate clipped).
#
# Injection splits are NOT affected -- they never call the tracker -- so only the
# six removal_az* splits are regenerated. The schedule stays `legacy`, i.e. v1
# semantics (attack on the final 8 frames of each scene) are unchanged; only the
# tracking is fixed.
#
# The previous splits are moved aside rather than deleted, so the old and new data
# can be compared before anything is discarded.
# NOTE: no `set -u` -- conda's activate scripts reference unbound vars and abort.
source /home/mellonzhang/miniconda3/etc/profile.d/conda.sh
conda activate atlas
export CUDA_HOME=/home/mellonzhang/miniconda3/envs/atlas
export PATH=$CUDA_HOME/bin:$PATH

AG=/home/mellonzhang/rsch/Polar-Hierarchical-Mamba/atlas_generation
export NUSCENES_ATLAS_ASSET_DIR=$AG/assets_trainval
CFG=../tools/cfgs/dataset_configs/nuscenes_dataset_trainval_gen.yaml
OUT=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/nuscenes_trainval/atlas_val
LOGDIR=$OUT/logs_regen_dt
mkdir -p "$LOGDIR"
cd "$AG"

STAMP=predt   # suffix for the pre-fix data we set aside

echo "===== regenerating v1 removal splits with dt=0.5 $(date +%H:%M:%S) ====="
pids=()
for az in 10 20 30 40 50 60; do
  # set the old data aside (never delete: we want to diff old vs new)
  for d in removal_az${az} removal_az${az}_annos; do
    if [ -d "$OUT/$d" ] && [ ! -d "$OUT/${d}_${STAMP}" ]; then
      mv "$OUT/$d" "$OUT/${d}_${STAMP}"
      echo "  archived $d -> ${d}_${STAMP}"
    fi
  done
done

for az in 10 20 30 40 50 60; do
  name=removal_az${az}
  ( python generate_data_removal_nuscenes.py --cfg_file $CFG \
      --save_dir_dataset $OUT/$name \
      --save_dir_annos   $OUT/${name}_annos \
      --az_width $az --log_file $LOGDIR/${name}.log --seed 42 \
      --schedule legacy \
      >$LOGDIR/${name}.out 2>&1 && echo "[ok] $name" || echo "[FAIL] $name" ) &
  pids+=($!)
done

echo "launched ${#pids[@]} jobs; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "===== DONE $(date +%H:%M:%S)  failures=$fail ====="
echo "new data : $OUT/removal_az{10..60}"
echo "pre-fix  : $OUT/removal_az{10..60}_${STAMP}  (kept for comparison)"
