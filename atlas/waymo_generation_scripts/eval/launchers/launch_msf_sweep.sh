#!/bin/bash
# MSF-4 / MSF-8 GT-sampling sweep on every 50-segment point-injection dataset, epochs MSF_EPOCHS (default
# "6 4 2": e6 first so every table gets its final-epoch number earliest; no-GT MSF comes from Delta later).
# One H100/L40S twin per (dataset, model) chaining:
#   seed_msf.sh <run>                    MSF proposal cache from OpenPCDet CP4F GT e36 (locked; no-op if complete)
#   eval_msf.sh <run> <4|8> <epochs>     the epochs whose summary.pkl is missing
# GT e6 lives in the flat msf{4,8}_cvpr/ dir (headline runs / linked into the views); other epochs in gt_epoch{E}/.
# Walltime = 1.3 x (1.3 h per MSF-4 / 2.2 h per MSF-8 checkpoint + 0.1 h seeding), measured on 50 segments
# (cvpr_msf{4,8}_tanhpatch on H100: 1.27 h / 2.19 h; 195-seg headline runs / 3.9: 0.95 h / 1.7 h).
# When the dataset's CP4F GT e36 eval is still queued (cvpr_ab_<tag>_cp4f36) the twin waits on it (afterany);
# its generation job (cvpr_gen_<run>) likewise.
#   bash launch_msf_sweep.sh [run ...]      DRY=1 ...
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/evals/launch_lib.sh"
SPL=(global_easy global_medium global_hard rel_fixed_easy rel_fixed_medium rel_fixed_hard)
ALL=()
for v in "" _att; do for s in "${SPL[@]}"; do ALL+=(sub50_${s}${v}_seed42); done; done
for v in "" _att; do for a in car_atlas_hard_global car_atlas_hard_flicker car_atlas_hard_randobj \
    cyc_atlas_global cyc_atlas_relative cyc_atlas_flicker cyc_atlas_randobj; do ALL+=(abl_${a}${v}_seed42); done; done
for c in car cyc; do for a in global relative flicker randobj; do ALL+=(abl_${c}_paste_${a}_seed42); done; done
for k in jpt jfr; do for s in "${SPL[@]}"; do ALL+=(${s}_att_${k}_50_seed42); done; done
for s in global_easy global_hard rel_fixed_easy rel_fixed_hard; do ALL+=(sub50_${s}_full_trace_seed42); done
RUNS=("$@"); [[ ${#RUNS[@]} -gt 0 ]] || RUNS=("${ALL[@]}")

EPW=${MSF_EPOCHS:-6 4 2}
ep_left () {   # RUN H -> wanted epochs without a summary.pkl (e6: flat dir; complete linked predictions count)
    local r=$1 h=$2 e out=()
    for e in $EPW; do
        local d=$ROOT/$r/evaluation/msf${h}_cvpr; [[ $e == 6 ]] || d=$d/gt_epoch$e
        [[ -f $d/summary.pkl ]] && continue
        if [[ $e == 6 && $(ls $d/predictions 2>/dev/null | grep -c '_p\.pkl$') -ge 50 ]]; then continue; fi
        out+=($e)
    done
    echo "${out[*]}"
}

n=0
for r in "${RUNS[@]}"; do
    [[ -e $ROOT/$r ]] || { echo "missing $r" >&2; continue; }
    e4=$(ep_left $r 4); e8=$(ep_left $r 8)
    [[ -z $e4 && -z $e8 ]] && { echo "DONE $r" >&2; continue; }
    tag=${r#abl_}; tag=${tag%_seed42}
    deps=()
    g=$(ids cvpr_ab_${tag}_cp4f36); [[ -n $g ]] && deps+=("afterany:$g")
    j=$(squeue -h -u "$USER" -n "cvpr_gen_$r" -o %i | head -1); [[ -n $j ]] && deps+=("afterok:$j")
    dep=$(IFS=,; echo "${deps[*]}")
    for h in 4 8; do
        [[ $h == 4 ]] && e=$e4 || e=$e8
        [[ -z $e ]] && continue
        per=$([[ $h == 4 ]] && echo 1.3 || echo 2.2); k=$(wc -w <<< "$e")
        mins=$(python3 -c "import math; print(math.ceil(1.3*($per*$k+0.1)*60/30)*30)")
        t=$(printf '%d:%02d:00' $((mins/60)) $((mins%60)))
        if [[ "${DRY:-0}" != 1 ]] && (( $(squeue -h -u "$USER" | wc -l) > ${QMAX:-470} )); then
            echo "STOP: queue above ${QMAX:-470}; rerun later for the rest" >&2; break 2; fi
        twins cvpr_msf${h}_${tag} $t "$dep" "seed_msf.sh|$r|||50" "eval_msf.sh|$r|$h $e||50" >/dev/null
    done
    n=$((n+1))
done
echo "MSF GT sweep planned/submitted for $n datasets"
