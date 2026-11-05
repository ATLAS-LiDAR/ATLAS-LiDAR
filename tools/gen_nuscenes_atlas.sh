#!/usr/bin/env bash
# Full nuScenes ATLAS generation on the trainval VAL split, resumable:
#   1. per-class trace libraries from TRAIN (car / pedestrian / cyclist), so the
#      attacker's library never contains a validation object
#   2. injection splits: inject_<global|relative>_<class>_<easy|medium|hard>
#   3. removal splits:   removal_az{10..60}
#
#   OUT       output root (required)
#   DATAROOT  nuScenes v1.0-trainval directory with samples/ and the info pkls
#             (default: data/nuscenes/v1.0-trainval in this repository)
#   ENV_BIN   optional bin/ directory of the Python environment to use
#
# Every step skips finished work (existing trace pkls, finished scenes), so on a
# preemptible queue just resubmit until it completes. Roughly 24 cores and 360 GB of
# memory with everything running in parallel; e.g.
#   sbatch -c 24 --mem=360G -t 08:00:00 --export=ALL,OUT=/data/atlas_nusc_val tools/gen_nuscenes_atlas.sh
# NOTE: no `set -u` -- conda's activate scripts reference unbound variables.
[ -n "$ENV_BIN" ] && export PATH=$ENV_BIN:$PATH
export OMP_NUM_THREADS=1
: "${OUT:?set OUT to the output directory}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AG=$REPO/atlas
export NUSCENES_ATLAS_ASSET_DIR=$AG/assets_trainval
TR=$NUSCENES_ATLAS_ASSET_DIR
CFG=$REPO/eval/cfgs/nuscenes_dataset_trainval_gen.yaml
DATAROOT=${DATAROOT:-$REPO/data/nuscenes/v1.0-trainval}
export OUT
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
echo "===== $(hostname) start $(date) ====="

# ---- 1. trace libraries --------------------------------------------------
need=()
for cls in car pedestrian cyclist; do
  [ -f "$TR/trace_${cls}_hard.pkl" ] || need+=($cls)
done
if [ ${#need[@]} -gt 0 ]; then
  echo ">>> building traces for: ${need[*]}"
  ( cd "$AG" && python -u build_nuscenes_traces_multiclass.py \
      --dataroot "$DATAROOT" --classes "${need[@]}" --out_dir "$TR" --seed 42 ) \
      > "$LOGDIR/traces.out" 2>&1 || { echo "trace build FAILED"; exit 1; }
  grep "^\[traces\]" "$LOGDIR/traces.out"
fi

classes=""
for cls in car pedestrian cyclist; do
  if [ -f "$TR/trace_${cls}_easy.pkl" ] && [ -f "$TR/trace_${cls}_medium.pkl" ] \
     && [ -f "$TR/trace_${cls}_hard.pkl" ]; then
    classes="$classes $cls"
  else
    echo "WARNING: no complete trace library for $cls, skipping its splits"
  fi
done

# ---- 2 + 3. injection and removal, all in parallel ------------------------
CLASSES="$classes" bash "$REPO/tools/run_multiclass_inject_nuscenes.sh" &
inj=$!

cd "$AG"
pids=()
for az in 10 20 30 40 50 60; do
  name=removal_az${az}
  ( python generate_data_removal_nuscenes.py --cfg_file "$CFG" \
      --save_dir_dataset "$OUT/$name" --save_dir_annos "$OUT/${name}_annos" \
      --az_width $az --log_file "$LOGDIR/${name}.log" --seed 42 \
      >"$LOGDIR/${name}.out" 2>&1 && echo "[ok] $name" || echo "[FAIL] $name" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
wait $inj

echo "===== DONE $(date) ====="
for d in "$OUT"/inject_* "$OUT"/removal_az*; do
  case "$d" in *_annos) continue;; esac
  echo "  $(basename "$d"): $(ls "$d"/*_d.pkl 2>/dev/null | wc -l)/150 scenes"
done
du -sh "$OUT"
