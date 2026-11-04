#!/bin/bash
# Post-rebuttal ablation evals (PTT/neurips NOTES_gt_vs_nogt_2026-10.md, abl_tables.md) on the 7
# regenerated ATLAS ablation sets (50 segs; same 50 segments as the rebuttal sets):
#   cars:     abl_car_atlas_hard_{global,flicker,randobj}_seed42   (global = new paired reference;
#             compare flicker/randobj against it, not against global_hard: see the NOTES TODO)
#   cyclists: abl_cyc_atlas_{global,flicker,randobj,relative}_seed42
# 28 checkpoints per set, as before: CP-SF e9/18/27/30 x {GT, no-GT}, CP4F e9/18/27/36 x {GT, no-GT},
# PTT GT e1-6 (CP4F GT e36 proposals) and no-GT e1-6 (CP4F no-GT e36 proposals).
# The 9 paste_* sets use real object points (correct intensities): their Oct 2026 evals stand.
# Layout per set (twins, NICE=5):
#   ab_<set>_cp4f36      CP4F GT + no-GT e36 (~2 h)      gate: PTT needs these proposals
#   ab_<set>_ptt_gt      PTT GT e1-6   (~3 h) --afterany gate
#   ab_<set>_ptt_nogt    PTT no-GT e1-6 (~3 h) --afterany gate   (+ cross-pairing if CROSS=1)
#   ab_<set>_cp4f_{gt,nogt}  CP4F e9/18/27 of one family (~3 h each), no dependency
#   ab_cpsf_<k>          CP-SF 8 checkpoints for 2 sets per job (~5.6 h), no dependency
# One gate per set (not one chain) so each set's PTT can start ~2 h after its gate starts.
#   bash launch_abl_evals.sh [set ...]      DRY=1 ...      CROSS=1 ...
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/evals/launch_lib.sh"

ALL=(abl_car_atlas_hard_global_seed42 abl_car_atlas_hard_flicker_seed42 abl_car_atlas_hard_randobj_seed42
     abl_cyc_atlas_global_seed42 abl_cyc_atlas_flicker_seed42 abl_cyc_atlas_randobj_seed42 abl_cyc_atlas_relative_seed42)
# VARIANT=_att: the same wave on abl_<name>_att_seed42 (generic registry variant); generation jobs that
# are still queued/running (cvpr_gen_<dataset>) become afterok dependencies instead of an error.
V=${VARIANT:-}
[[ -n $V ]] && ALL=("${ALL[@]/%_seed42/${V}_seed42}")
SETS=("$@"); [[ ${#SETS[@]} -gt 0 ]] || SETS=("${ALL[@]}")
declare -A GD
for r in "${SETS[@]}"; do GD[$r]=$(gen_dep_of $r 50); done

# PTT epochs (CVPR 50-seg plan: 2 4 6). Walltime 1.5 x measured (6 epochs: 3.05 h max -> 0.51 h/epoch)
PTT_E=${PTT_EPOCHS:-2 4 6}
PTT_T=$(python3 -c "import math; m=math.ceil(1.5*0.51*$(wc -w <<< "$PTT_E")*60/30)*30+30; print(f'{m//60}:{m%60:02d}:00')")
CPSF_L="gt_epoch9 nogt_epoch9 gt_epoch18 nogt_epoch18 gt_epoch27 nogt_epoch27 gt_epoch30 nogt_epoch30"
add () { task_done "$1" "$3" "$5" || T+=("$2|$1|$4|$5|$6"); }
tag_of () { local t=${1#abl_}; echo "${t%_seed42}"; }   # keeps the variant infix

for r in "${SETS[@]}"; do
    tag=$(tag_of $r)
    T=(); add $r eval_cp4f.sh cp4f_atlas all "gt_epoch36 nogt_epoch36" 50
    gate=""; [[ ${#T[@]} -gt 0 ]] && gate=$(twins cvpr_ab_${tag}_cp4f36 03:00:00 "${GD[$r]}" "${T[@]}")
    ga=""; [[ -n $gate ]] && ga="afterany:$gate"
    T=(); add $r eval_ptt.sh ptt32_atlas gt "$(for e in $PTT_E; do printf 'gt_epoch%s ' $e; done)" 50
    [[ ${#T[@]} -gt 0 ]] && twins cvpr_ab_${tag}_ptt_gt $PTT_T "$ga" "${T[@]}" >/dev/null
    T=(); add $r eval_ptt.sh ptt32_atlas nogt "$(for e in $PTT_E; do printf 'nogt_epoch%s ' $e; done)" 50
    [[ "${CROSS:-0}" == 1 ]] && add $r eval_ptt.sh ptt32_atlas cross "gtptt_nogtprop_epoch6 nogtptt_gtprop_epoch6" 50
    tl=$PTT_T; [[ "${CROSS:-0}" == 1 ]] && tl=06:00:00
    [[ ${#T[@]} -gt 0 ]] && twins cvpr_ab_${tag}_ptt_nogt $tl "$ga" "${T[@]}" >/dev/null
    for fam in gt nogt; do
        T=(); add $r eval_cp4f.sh cp4f_atlas $fam "${fam}_epoch9 ${fam}_epoch18 ${fam}_epoch27" 50
        [[ ${#T[@]} -gt 0 ]] && twins cvpr_ab_${tag}_cp4f_$fam 04:00:00 "${GD[$r]}" "${T[@]}" >/dev/null
    done
done

# CP-SF: two sets per chain; the job is named after its first set so reruns keep stable names
for ((i = 0; i < ${#SETS[@]}; i += 2)); do
    T=(); dd=(); for r in "${SETS[@]:i:2}"; do add $r eval_cpsf.sh cp_sf_atlas "" "$CPSF_L" 50; [[ -n ${GD[$r]} ]] && dd+=(${GD[$r]#afterok:}); done
    d=""; [[ ${#dd[@]} -gt 0 ]] && d="afterok:$(IFS=:; echo "${dd[*]}")"
    [[ ${#T[@]} -gt 0 ]] && twins cvpr_ab_cpsf_$(tag_of ${SETS[i]}) 07:30:00 "$d" "${T[@]}" >/dev/null
done
echo "ablation wave planned/submitted for ${#SETS[@]} sets (NICE=$NICE, CROSS=${CROSS:-0})"
