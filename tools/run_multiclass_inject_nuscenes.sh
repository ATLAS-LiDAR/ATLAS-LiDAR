#!/usr/bin/env bash
# Phantom PEDESTRIAN / CYCLIST injection splits on the nuScenes trainval VAL split.
#
# Same generators and settings as atlas/scripts/run_all_variants_trainval.sh
# (seed 42, legacy schedule); only the trace library changes, to the per-class
# libraries built by atlas/build_nuscenes_traces_multiclass.py. Output directories
# are named inject_<mode>_<class>_<diff>, next to the car splits.
#
#   OUT      output root (required)
#   CLASSES  default "pedestrian cyclist"
#   EXTRA    extra generator flags, e.g. "--schedule continuous"
#   ENV_BIN  optional bin/ directory of the Python environment to use
#
# Every split is launched in parallel; run it on a machine with enough cores and memory.
# NOTE: no `set -u` -- conda's activate scripts reference unbound variables.
[ -n "$ENV_BIN" ] && export PATH=$ENV_BIN:$PATH
export OMP_NUM_THREADS=1
: "${OUT:?set OUT to the output directory}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AG=$REPO/atlas
export NUSCENES_ATLAS_ASSET_DIR=$AG/assets_trainval
CFG=$REPO/eval/cfgs/nuscenes_dataset_trainval_gen.yaml
TR=$NUSCENES_ATLAS_ASSET_DIR
CLASSES=${CLASSES:-"pedestrian cyclist"}
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
cd "$AG"

for cls in $CLASSES; do
  for diff in easy medium hard; do
    [ -f "$TR/trace_${cls}_${diff}.pkl" ] || {
      echo "missing $TR/trace_${cls}_${diff}.pkl -- run build_nuscenes_traces_multiclass.py first"; exit 1; }
  done
done

pids=()
echo "===== launching $(date +%H:%M:%S) ====="
for cls in $CLASSES; do
  for mode in global relative; do
    for diff in easy medium hard; do
      name=inject_${mode}_${cls}_${diff}
      ( python generate_data_${mode}_nuscenes.py --cfg_file "$CFG" \
          --save_dir_dataset "$OUT/$name" \
          --save_dir_annos   "$OUT/${name}_annos" \
          --trace_file "$TR/trace_${cls}_${diff}.pkl" \
          --log_file "$LOGDIR/${name}.log" --seed 42 $EXTRA \
          >"$LOGDIR/${name}.out" 2>&1 && echo "[ok] $name" \
          || echo "[FAIL] $name" ) &
      pids+=($!)
    done
  done
done

echo "waiting on ${#pids[@]} jobs..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "===== DONE $(date +%H:%M:%S)  failures=$fail ====="
for cls in $CLASSES; do
  for d in "$OUT"/inject_*_${cls}_*; do
    case "$d" in *_annos) continue;; esac
    echo "  $(basename "$d"): $(ls "$d"/*_d.pkl 2>/dev/null | wc -l) scenes"
  done
done
