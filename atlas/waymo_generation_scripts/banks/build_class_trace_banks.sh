#!/bin/bash
# Trace banks for the class-wise injection table (NEW_DATASETS.md): pedestrian easy/medium/hard and cyclist
# easy/medium. Same builder, source segments and selection rule as the existing cyclist bank
# (build_val_object_bank.sh): val segments EXCEPT the 50 attacked ones, 300 traces per bin, one instance per
# distinct object first. Bins by points on the object (all ranges <= 60 m), shared by both classes so the
# cyclist "hard" bin equals the existing data/traces_val_cyclist_hard.pkl band (100-300 pts):
#   easy 30-60, medium 60-100, hard 100-300
# The builder prints candidates per bin and the point percentiles at 13-22 m (where ATLAS places phantoms) in
# bank_summary.json; if a bin has very few distinct objects, widen it and rename the datasets' trace files.
#   bash banks/build_class_trace_banks.sh           (CPU, ~8 cores, 128 GB, a few hours; inside the pcdet env)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../config.sh"
EX=$(mktemp -d); trap 'rm -rf "$EX"' EXIT
while read -r s; do [[ -n $s ]] && : > "$EX/$s.pkl"; done < "$DATA_DIR/segments_50_seed42.txt"
cd "$OPENPCDET"; export PYTHONUNBUFFERED=1
COMMON=(--openpcdet "$OPENPCDET" --cfg "$CP_SF_CFG" --data_path "$WAYMO_ROOT" --exclude_dataset_dir "$EX"
        --max_range 60 --min_points 5 --seed 42)
[[ -f $BANK_DIR/pedestrian/traces_val_pedestrian_hard.pkl ]] || \
python "$HERE/build_val_object_bank.py" "${COMMON[@]}" --out_dir "$BANK_DIR/pedestrian" \
    --classes Pedestrian --frame_stride 5 \
    --trace_band Pedestrian:easy:30:60:300 --trace_band Pedestrian:medium:60:100:300 --trace_band Pedestrian:hard:100:300:300
[[ -f $BANK_DIR/cyclist3/traces_val_cyclist_medium.pkl ]] || \
python "$HERE/build_val_object_bank.py" "${COMMON[@]}" --out_dir "$BANK_DIR/cyclist3" \
    --classes Cyclist --frame_stride 1 \
    --trace_band Cyclist:easy:30:60:300 --trace_band Cyclist:medium:60:100:300 --trace_band Cyclist:hard:100:300:300
python - "$BANK_DIR" <<'PY'
import json, sys
for d in ("pedestrian", "cyclist3"):
    s = json.load(open(f"{sys.argv[1]}/{d}/bank_summary.json"))
    print(d, json.dumps({k: v for k, v in s.get("trace_banks", {}).items()}, indent=1))
    print(d, "13-22 m:", json.dumps(s["classes"]))
PY
echo "Done. Cyclist hard for the class-wise table stays data/traces_val_cyclist_hard.pkl (existing abl_cyc_atlas_* sets)."
