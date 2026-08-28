#!/usr/bin/env bash
# Generate the 24 SCHEDULE variants of the ATLAS adversarial splits on the
# nuScenes trainval VAL split, in parallel.
#
# Difference from run_all_variants_trainval.sh: instead of the legacy alternating
# 32-frame window (which on ~40-frame nuScenes scenes degenerates to a single
# 8-frame attack pinned to the tail), each scene gets exactly ONE attack whose
# length L varies per scene (2-32 frames = 1-16 s at 2 Hz), in two continuity
# regimes:
#   _cont     : L contiguous attacked frames
#   _intermit : the SAME attack flickering on/off over a wider span, with the
#               attacked-frame count held equal to _cont at the same L
#
# L and the placement are keyed to the scene name, so they are IDENTICAL across
# all 12 base variants and both regimes -- that is what makes removal-vs-injection
# a paired per-sequence comparison.
#
# Output goes to a SEPARATE root (atlas_val_sched/) so the existing 12 splits and
# their results CSV are untouched.
# NOTE: no `set -u` -- conda's activate scripts reference unbound vars and abort.
source /home/mellonzhang/miniconda3/etc/profile.d/conda.sh
conda activate atlas
export CUDA_HOME=/home/mellonzhang/miniconda3/envs/atlas
export PATH=$CUDA_HOME/bin:$PATH

AG=/home/mellonzhang/rsch/Polar-Hierarchical-Mamba/atlas_generation
export NUSCENES_ATLAS_ASSET_DIR=$AG/assets_trainval
CFG=../tools/cfgs/dataset_configs/nuscenes_dataset_trainval_gen.yaml
OUT=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/nuscenes_trainval/atlas_val_sched
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
cd "$AG"
TR=$AG/assets_trainval
SEED=42
SCHED_SEED=42
MINL=2
MAXL=32
DUTY=0.5

pids=()

echo "===== launching 24 schedule variants in parallel $(date +%H:%M:%S) ====="
for sched in continuous intermittent; do
  case $sched in
    continuous)   sfx=cont ;;
    intermittent) sfx=intermit ;;
  esac

  # ---- injection: global / relative x easy / medium / hard
  for mode in global relative; do
    for diff in easy medium hard; do
      name=inject_${mode}_${diff}_${sfx}
      ( python generate_data_${mode}_nuscenes.py --cfg_file $CFG \
          --save_dir_dataset $OUT/$name \
          --save_dir_annos   $OUT/${name}_annos \
          --trace_file $TR/trace_${diff}.pkl \
          --log_file $LOGDIR/${name}.log --seed $SEED \
          --schedule $sched --sched_seed $SCHED_SEED \
          --min_len $MINL --max_len $MAXL --duty $DUTY \
          >$LOGDIR/${name}.out 2>&1 && echo "[ok] $name" \
          || echo "[FAIL] $name" ) &
      pids+=($!)
    done
  done

  # ---- removal: az 10..60
  for az in 10 20 30 40 50 60; do
    name=removal_az${az}_${sfx}
    ( python generate_data_removal_nuscenes.py --cfg_file $CFG \
        --save_dir_dataset $OUT/$name \
        --save_dir_annos   $OUT/${name}_annos \
        --az_width $az \
        --log_file $LOGDIR/${name}.log --seed $SEED \
        --schedule $sched --sched_seed $SCHED_SEED \
        --min_len $MINL --max_len $MAXL --duty $DUTY \
        >$LOGDIR/${name}.out 2>&1 && echo "[ok] $name" \
        || echo "[FAIL] $name" ) &
    pids+=($!)
  done
done

echo "launched ${#pids[@]} jobs; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "===== DONE $(date +%H:%M:%S)  failures=$fail ====="
echo "output: $OUT"
