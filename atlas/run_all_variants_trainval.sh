#!/usr/bin/env bash
# Generate all 12 ATLAS adversarial variants on the nuScenes trainval VAL split,
# IN PARALLEL. Uses the train-only trace library + assets in assets_trainval/
# (no leakage), selected via NUSCENES_ATLAS_ASSET_DIR.
# NOTE: no `set -u` — conda's activate scripts reference unbound vars and abort.
source /home/mellonzhang/miniconda3/etc/profile.d/conda.sh
conda activate atlas
export CUDA_HOME=/home/mellonzhang/miniconda3/envs/atlas
export PATH=$CUDA_HOME/bin:$PATH

AG=/home/mellonzhang/rsch/Polar-Hierarchical-Mamba/atlas_generation
export NUSCENES_ATLAS_ASSET_DIR=$AG/assets_trainval
CFG=../tools/cfgs/dataset_configs/nuscenes_dataset_trainval_gen.yaml
OUT=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/nuscenes_trainval/atlas_val
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
cd "$AG"
TR=$AG/assets_trainval

pids=()
launch() { echo ">>> launch $1 $(date +%H:%M:%S)"; shift; "$@" & pids+=($!); }

echo "===== launching 12 variants in parallel $(date +%H:%M:%S) ====="
for mode in global relative; do
  for diff in easy medium hard; do
    ( python generate_data_${mode}_nuscenes.py --cfg_file $CFG \
        --save_dir_dataset $OUT/inject_${mode}_${diff} \
        --save_dir_annos   $OUT/inject_${mode}_${diff}_annos \
        --trace_file $TR/trace_${diff}.pkl \
        --log_file $LOGDIR/${mode}_${diff}.log --seed 42 \
        >$LOGDIR/${mode}_${diff}.out 2>&1 && echo "[ok] inject_${mode}_${diff}" \
        || echo "[FAIL] inject_${mode}_${diff}" ) &
    pids+=($!)
  done
done
for az in 10 20 30 40 50 60; do
  ( python generate_data_removal_nuscenes.py --cfg_file $CFG \
      --save_dir_dataset $OUT/removal_az${az} \
      --save_dir_annos   $OUT/removal_az${az}_annos \
      --az_width $az --log_file $LOGDIR/removal_az${az}.log --seed 42 \
      >$LOGDIR/removal_az${az}.out 2>&1 && echo "[ok] removal_az${az}" \
      || echo "[FAIL] removal_az${az}" ) &
  pids+=($!)
done

echo "waiting on ${#pids[@]} jobs..."
wait
echo "===== ALL VARIANTS DONE $(date +%H:%M:%S) ====="
for d in $OUT/inject_* $OUT/removal_az*; do
  case "$d" in *_annos) continue;; esac
  echo "  $(basename $d): $(ls $d/*_d.pkl 2>/dev/null | wc -l) scenes"
done
