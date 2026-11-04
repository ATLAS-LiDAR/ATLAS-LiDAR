#!/bin/bash
# Final-epoch GT-model evals of the point-removal / point-loss-baseline / Gaussian-control sets on the 50 segments
# (CVPR plan: removal is not a GT-vs-no-GT study). Per set, two H100/L40S twins:
#   rm_<set>_cp4f      CP4F GT e36 (OpenPCDet; also the PTT/MSF proposals)          ~1.0 h -> 2:00
#   rm_<set>_rest      CP-SF e30, PTT GT e6, seed MSF cache, MSF-4 e6, MSF-8 e6       ~4.5 h -> 6:00 (afterany cp4f)
# Inference uses the CVPR eval scripts (same as injection); SCORING is the rebuttal removal scorer
# (PTT/atlas_removal_eval_only.py, target hidden / active frames), run afterwards on CPU (score_removal step).
# Verified 2026-11-04 on a 1-segment az20 view: all five pipelines run on removal data; CP-SF predictions equal the
# July CP-SF predictions up to GPU nondeterminism in the low-score tail.
#   bash launch_removal_evals.sh [set ...]      DRY=1 ...
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/evals/launch_lib.sh"
ALL=(); for w in 10 20 30 40 50 60; do ALL+=(atlas_removal_az${w}_seed42); done
ALL+=(baseline_uniform_50_seed42 baseline_uniform_90_seed42 baseline_ring_16_seed42 baseline_ring_32_seed42
      baseline_matched_az20_seed42 baseline_matched_az50_seed42 gaussian_sigma_008_global_easy_seed42)
SETS=("$@"); [[ ${#SETS[@]} -gt 0 ]] || SETS=("${ALL[@]}")
for r in "${SETS[@]}"; do
    need_complete $r 50
    tag=${r%_seed42}
    T=(); task_done $r cp4f_atlas "gt_epoch36" || T+=("eval_cp4f.sh|$r|gt|gt_epoch36|50")
    g=""; [[ ${#T[@]} -gt 0 ]] && g=$(twins cvpr_rm_${tag}_cp4f 02:00:00 "" "${T[@]}")
    d=""; [[ -n $g ]] && d="afterany:$g"
    T=()
    task_done $r cp_sf_atlas "gt_epoch30" || T+=("eval_cpsf.sh|$r||gt_epoch30|50")
    task_done $r ptt32_atlas "gt_epoch6" || T+=("eval_ptt.sh|$r|gt|gt_epoch6|50")
    T+=("seed_msf.sh|$r|||50")
    task_done $r msf4_cvpr "" || T+=("eval_msf.sh|$r|4 6||50")
    task_done $r msf8_cvpr "" || T+=("eval_msf.sh|$r|8 6||50")
    [[ ${#T[@]} -gt 1 ]] && twins cvpr_rm_${tag}_rest 06:00:00 "$d" "${T[@]}" >/dev/null
done
echo "removal/baseline final-epoch evals planned/submitted for ${#SETS[@]} sets"
