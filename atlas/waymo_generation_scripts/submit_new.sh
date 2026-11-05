#!/bin/bash
# Submit every new dataset of NEW_DATASETS.md as CPU Slurm jobs, with the right dependencies:
#   class trace banks  -> pedestrian / cyclist easy-medium sets
#   paired reference   -> scene-level sets (spray, crosstalk, fog, temporal faults, pose noise)
#   bash submit_new.sh                 all new datasets           DRY=1 bash submit_new.sh   (print only)
#   bash submit_new.sh <name> ...      only these (names from: bash generate.sh --list-new)
# Each job runs `bash generate.sh <name>` (resumable: rerunning skips finished segments, then verifies).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/datasets.sh"
[[ $SLURM_ACCOUNT == CHANGE_ME ]] && { echo "set SLURM_ACCOUNT in config.sh" >&2; exit 1; }
LOG=$OUT_ROOT/_logs; mkdir -p "$LOG"
ACT=""; [[ -n $CONDA_ENV ]] && ACT="source activate $CONDA_ENV;"
sub () {   # NAME HOURS MEM_GB CPUS DEP CMD
    local d=(); [[ -n $5 ]] && d=(--dependency=afterok:$5)
    if [[ ${DRY:-0} == 1 ]]; then echo "DRY $1 ${2}h ${3}G ${4}c dep=${5:-none} :: $6" >&2; echo 0; return; fi
    sbatch --parsable --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION --nodes=1 --ntasks=1 \
        --cpus-per-task=$4 --mem=${3}G --time=$2:00:00 --job-name=$1 "${d[@]}" \
        --output="$LOG/%x_%j.out" --error="$LOG/%x_%j.err" --wrap="set -e; $ACT $6"
}
DS=("$@"); [[ ${#DS[@]} -gt 0 ]] || mapfile -t DS < <(new_datasets)
BANK=""; REF=""
need_bank=0; need_ref=0
for d in "${DS[@]}"; do [[ $d == abl_ped_* || $d == abl_cyc_atlas_*_easy* || $d == abl_cyc_atlas_*_medium* ]] && need_bank=1; spec $d; [[ $GEN == scene ]] && need_ref=1; done
if [[ $need_bank == 1 && ! -f $BANK_DIR/pedestrian/traces_val_pedestrian_hard.pkl || $need_bank == 1 && ! -f $BANK_DIR/cyclist3/traces_val_cyclist_medium.pkl ]]; then
    BANK=$(sub atlas_bank_classes 12 128 8 "" "bash $HERE/banks/build_class_trace_banks.sh"); echo "bank job $BANK"
fi
R0=${SCENE_REF:-atlas_removal_az20_seed42}
if [[ $need_ref == 1 && ! -f $OUT_ROOT/$R0/metrics/run_manifest.json ]]; then
    # removal generator checks pairing against baseline_matched_az50 -> generate that first
    B=""; [[ -f $OUT_ROOT/baseline_matched_az50_seed42/metrics/run_manifest.json ]] || \
        B=$(sub gen_baseline_matched_az50_seed42 4 96 8 "" "bash $HERE/generate.sh baseline_matched_az50_seed42")
    REF=$(sub gen_$R0 4 96 8 "$B" "bash $HERE/generate.sh $R0"); echo "reference job $REF (after ${B:-none})"
fi
for d in "${DS[@]}"; do
    spec $d
    dep=""; t=4; mem=96
    case $GEN in
      relative) t=6; mem=160 ;;
      global)   t=4; mem=96 ;;
      scene)    t=4; mem=64; dep=$REF ;;
    esac
    [[ $d == abl_ped_* || $d == abl_cyc_atlas_*_easy* || $d == abl_cyc_atlas_*_medium* ]] && dep=$BANK
    j=$(sub gen_$d $t $mem 8 "$dep" "bash $HERE/generate.sh $d"); echo "$d: $j (${t}h, dep ${dep:-none})"
done
