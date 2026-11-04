#!/bin/bash
# Chain task (run_chain.sh): seed <run>/evaluation/msf_cp4f_proposals/gt_epoch36 from the run's OpenPCDet CP4F GT e36
# predictions (seed_msf_proposals.py: frame_id alignment + clean full-val gate). No-op when the cache is complete.
#   bash seed_msf.sh <run>
set -euo pipefail
C=/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/cvpr_scripts
ROOT=${GEN_ROOT:-/storage/scratch1/9/spanse30/cvpr_datasets}
FULLVAL_GT=/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/val/default/result_fixed.pkl
r=${1:?usage: seed_msf.sh <run>}
out=$ROOT/$r/evaluation/msf_cp4f_proposals/gt_epoch36
n=$(ls "$ROOT/$r/dataset" | grep -c '\.pkl$')
# MSF-4 and MSF-8 jobs of a run may start together: one seeds under a lock, the other waits for it
lock=$ROOT/$r/evaluation/.seed_msf.lock; mkdir -p "$ROOT/$r/evaluation"
until mkdir "$lock" 2>/dev/null; do
    [[ -n $(find "$lock" -maxdepth 0 -mmin +120 2>/dev/null) ]] && { echo "seed_msf: stale lock removed"; rmdir "$lock"; continue; }
    echo "seed_msf: $r waiting for lock"; sleep 30; done
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
have=$(ls "$out" 2>/dev/null | grep -c '_cp4f\.pkl$' || true)
if [[ $have -eq $n ]]; then echo "seed_msf: $r cache complete ($have)"; exit 0; fi
p=$(ls "$ROOT/$r/evaluation/cp4f_atlas/gt_epoch36/predictions" 2>/dev/null | grep -c '_p\.pkl$' || true)
[[ $p -eq $n ]] || { echo "seed_msf: $r has $p/$n CP4F GT e36 predictions" >&2; exit 1; }
module load anaconda3; eval "$(conda shell.bash hook)"; conda activate /storage/project/r-gchou3-0/spanse30/ptt_env
python -I "$C/seed_msf_proposals.py" "$ROOT/$r" "$out" --label gt_epoch36 --fullval "$FULLVAL_GT"
