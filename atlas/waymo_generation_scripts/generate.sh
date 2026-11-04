#!/bin/bash
# Generate one or more registered datasets (CPU only; no model is run).
#   bash generate.sh <dataset> [dataset ...]       e.g. bash generate.sh global_hard_att_seed42
#   bash generate.sh --list                         every registered name
# Segments already written are skipped by the generators, so rerunning resumes. Run inside the env that has
# OpenPCDet (pcdet), numpy, torch, matplotlib and plotly. Typical cost per 50-segment dataset (8 cores):
# injection ~2 h global / ~3.3 h relative (peak RAM ~56 / ~132 GB); removal and baselines ~1.3 h.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/datasets.sh"
[[ ${1:-} == --list ]] && { all_datasets; exit 0; }
[[ $# -ge 1 ]] || { echo "usage: $0 <dataset> ... | --list" >&2; exit 1; }
export PYTHONPATH="$GEN_DIR:$OPENPCDET:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "$DATA_DIR"          # generators read waymo_top_lidar_inclinations.npy / range_conditioned_stats.npy from cwd

for d in "$@"; do
    spec "$d"
    R=$OUT_ROOT/$d; mkdir -p "$R"/{dataset,annos,metrics}
    for f in "${ARGS[@]}"; do [[ $f == /*.pkl ]] && { [[ -f $f ]] || { echo "missing bank file $f (README step 0)" >&2; exit 1; }; }; done
    [[ -n $NOTE ]] && echo "note: $NOTE"
    echo "=== $(date '+%F %T')  $d  [$GEN, intensity=$IMODE, jitter=$JIT_IN/$JIT_INTER m] -> $R"
    export ATLAS_INTENSITY_MODE=$IMODE ATLAS_JITTER_INNER=$JIT_IN ATLAS_JITTER_INTER=$JIT_INTER
    if [[ $IMODE == attenuated ]]; then export ATLAS_INTENSITY_VALUE=$ATT_I; else unset ATLAS_INTENSITY_VALUE; fi
    common=(--save_dir_dataset "$R/dataset" --save_dir_annos "$R/annos" --metrics_dir "$R/metrics"
            --log_file "$R/generation.log" --variant_name "$d" --seed 42 --data_path "$WAYMO_ROOT")
    case $GEN in
      global|relative)
        python -u "$GEN_DIR/gen_atlas_${GEN}_cvpr.py" --cp_mf_root "$OPENPCDET" --cp_mf_cfg "$CP_MF_CFG" \
            --cp_sf_cfg "$CP_SF_CFG" "${common[@]}" "${ARGS[@]}" ;;
      baseline)
        python -u "$GEN_DIR/generate_data_adverse_baselines.py" --cp_mf_root "$OPENPCDET" --cp_mf_cfg "$CP_MF_CFG" \
            --cp_sf_cfg "$CP_SF_CFG" "${common[@]}" --num_sequences ${NSEQ:-50} --sequence_selection ${SELECTION:-random} \
            --pairing_box_atol 1e-5 "${ARGS[@]}" ;;
      removal)
        python -u "$GEN_DIR/gen_removal_cvpr.py" --cp_mf_root "$OPENPCDET" --cp_mf_cfg "$CP_MF_CFG_RM" \
            --cp_sf_cfg "$CP_SF_CFG_RM" "${common[@]}" --num_sequences ${NSEQ:-50} --sequence_selection ${SELECTION:-random} "${ARGS[@]}" ;;
    esac
    python -I "$HERE/verify_dataset.py" "$R" --expect ${NSEQ:-50}
done
