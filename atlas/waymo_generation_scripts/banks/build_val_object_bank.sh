#!/bin/bash
# Val object banks for the paste and cyclist ablations (see build_val_object_bank.py). CPU, ~8 cores, 128 GB, a few h.
#   bash build_val_object_bank.sh           (paths from ../config.sh; run inside the pcdet env)
# Sources: val segments EXCEPT the 50 attacked segments (data/segments_50_seed42.txt), so a pasted object never
# comes from the scene it is pasted into.
#   $BANK_DIR/vehicle/  every 5th frame -> object_bank_val.pkl (paste, Vehicle), traces_val_vehicle_hard.pkl
#   $BANK_DIR/cyclist/  every frame     -> object_bank_val.pkl (paste, Cyclist),
#                                          traces_val_cyclist_{easy,hard}.pkl (30-100 / 100-300 pts; ATLAS cyclist
#                                          ablations use traces_val_cyclist_hard.pkl)
# Cyclists are rare in Waymo, hence stride 1 for them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BANKS_SRC="$HERE"   # config.sh overwrites HERE with its own dir
source "$HERE/../config.sh"
# the builder excludes the segments whose <segment>.pkl exists in --exclude_dataset_dir: make that list from the file
EX=$(mktemp -d); trap 'rm -rf "$EX"' EXIT
while read -r s; do [[ -n $s ]] && : > "$EX/$s.pkl"; done < "$DATA_DIR/segments_50_seed42.txt"
cd "$OPENPCDET"; export PYTHONUNBUFFERED=1
COMMON=(--openpcdet "$OPENPCDET" --cfg "$CP_SF_CFG" --data_path "$WAYMO_ROOT" --exclude_dataset_dir "$EX"
        --max_range 60 --min_points 5 --seed 42)
python "$BANKS_SRC/build_val_object_bank.py" "${COMMON[@]}" --out_dir "$BANK_DIR/vehicle" \
    --classes Vehicle --frame_stride 5 --trace_band Vehicle:hard:500:1000:300
python "$BANKS_SRC/build_val_object_bank.py" "${COMMON[@]}" --out_dir "$BANK_DIR/cyclist" \
    --classes Cyclist --frame_stride 1 \
    --trace_band Cyclist:easy:30:100:300 --trace_band Cyclist:hard:100:300:300
echo Done.
