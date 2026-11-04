#!/bin/bash
# Runs one or more eval tasks SEQUENTIALLY inside one Slurm allocation (a "chain"). Submitted as
# H100 + L40S twins by the launchers (RACE_KEY): the first copy to start on a working GPU takes the
# lock and cancels its twin. Each task is one argument:
#     "<script>|<run>|<extra args (space-separated)>|<LABELS>|<EXPECT_SEGS>"
# e.g. "eval_cpsf.sh|global_easy_full_seed42||gt_epoch30|195"
# Tasks run with RACE_KEY unset (the chain already owns the race). A failing task does not stop the
# chain; the job exits non-zero if any task failed (dependents use afterany + their own input checks).
set -uo pipefail
E=/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/cvpr_scripts/evals
export PATH="/opt/slurm/current/bin:$PATH"
[[ $# -ge 1 ]] || { echo "usage: $0 '<script>|<run>|<args>|<LABELS>|<EXPECT_SEGS>' ..." >&2; exit 2; }

if [[ -n "${RACE_KEY:-}" ]]; then
    /storage/project/r-gchou3-0/spanse30/ptt_env/bin/python -I -c "import torch; torch.ones(1024, device='cuda').sum().item()" >/dev/null 2>&1 \
        || { echo "GPU check failed on $(hostname -s); leaving the race" >&2; exit 1; }
    LOCK=/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/PTT/neurips/race_locks/$RACE_KEY
    mkdir -p "$(dirname "$LOCK")"
    mkdir "$LOCK" 2>/dev/null || { echo "race: $RACE_KEY claimed by $(cat "$LOCK/job" 2>/dev/null); exiting"; exit 0; }
    echo "$SLURM_JOB_ID $(hostname -s) ${SLURM_JOB_PARTITION:-}" > "$LOCK/job"
    for j in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -o %i); do
        [[ "$j" != "$SLURM_JOB_ID" ]] && scancel "$j" && echo "race: won on ${SLURM_JOB_PARTITION:-?}, cancelled twin $j"
    done
fi

fail=0
for t in "$@"; do
    IFS='|' read -r script run args labels expect <<< "$t"
    echo "=================== $(date '+%F %T')  $script $run $args  LABELS=[$labels] EXPECT_SEGS=$expect"
    # shellcheck disable=SC2086
    env -u RACE_KEY LABELS="$labels" EXPECT_SEGS="$expect" bash "$E/$script" "$run" $args
    rc=$?
    echo "=================== $(date '+%F %T')  exit $rc: $script $run"
    [[ $rc -eq 0 ]] || fail=1
done
exit $fail
