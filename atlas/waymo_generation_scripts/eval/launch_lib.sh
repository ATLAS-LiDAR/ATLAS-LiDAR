# Shared helpers for the CVPR eval launchers (source this file; do not run it).
#   twins NAME TIME DEP TASK...      H100 + L40S twins of evals/run_chain.sh; prints "h100id:l40sid"
#   cpu_job NAME TIME MEM DEP CMD    one CPU job (gts-gchou3, cpu-small/medium); prints its id
#   task_done RUN DETSUB "LABELS"    all LABELS of RUN/evaluation/DETSUB have summary.pkl
#   ids NAME                         queued/running job ids with that name, colon-joined
# Accounts: H100 = gts-gchou3-ideasci23_dgx, L40S = gts-gchou3-ideas_l40s, CPU = gts-gchou3.
# Paid accounts are never used. NICE (default 5) ranks these jobs below the main-paper wave.
# A stale race lock (left by a finished/failed twin of the same name) is removed before
# submitting, so a resubmitted job does not exit as "claimed". DRY=1 prints instead of submitting.
export PATH="/opt/slurm/current/bin:$PATH"
C=/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/cvpr_scripts
E=$C/evals
ROOT=${GEN_ROOT:-/storage/scratch1/9/spanse30/cvpr_datasets}
LOG=$C/logs/evals; mkdir -p "$LOG"
LOCKS=/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/PTT/neurips/race_locks
NICE=${NICE:-5}
H100=(--account=gts-gchou3-ideasci23_dgx --partition=gpu-h100 --gres=gpu:h100:1)
L40S=(--account=gts-gchou3-ideas_l40s --partition=gpu-l40s --gres=gpu:l40s:1)
GPUJOB=(--cpus-per-task=8 --mem=160G --nodes=1 --ntasks=1)
ENV=/storage/project/r-gchou3-0/spanse30/ptt_env
FULLVAL_GT=/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/val/default/result_fixed.pkl

ids () { squeue -h -u "$USER" -n "$1" -o %i | paste -sd: ; }
queued () { [[ -n "$(ids "$1")" ]]; }

task_done () {           # RUN DETSUB "LABELS" (MSF: DETSUB=msf4_cvpr, LABELS="")
    local run=$1 sub=$2 l
    if [[ -z "$3" ]]; then [[ -f "$ROOT/$run/evaluation/$sub/summary.pkl" ]]; return; fi
    for l in $3; do [[ -f "$ROOT/$run/evaluation/$sub/$l/summary.pkl" ]] || return 1; done
}

twins () {               # NAME TIME DEP TASK...   (DEP: "" or afterany:/afterok: ids)
    local name=$1 t=$2 dep=$3; shift 3
    if queued "$name"; then echo "SKIP $name (queued/running $(ids "$name"))" >&2; ids "$name"; return; fi
    local d=(); [[ -n $dep ]] && d=(--dependency=$dep)
    if [[ "${DRY:-0}" == 1 ]]; then echo "DRY  $name  $t  ${dep:-no-dep}  :: $*" >&2; echo "0:0"; return; fi
    [[ -d "$LOCKS/$name" ]] && rm -rf "${LOCKS:?}/${name:?}"
    local com=(--parsable --job-name=$name --time=$t --nice=$NICE "${GPUJOB[@]}" "${d[@]}"
               --export=ALL,RACE_KEY=$name --output="$LOG/%x_%j.out" --error="$LOG/%x_%j.err")
    local a b; a=$(sbatch "${com[@]}" "${H100[@]}" "$E/run_chain.sh" "$@"); b=$(sbatch "${com[@]}" "${L40S[@]}" "$E/run_chain.sh" "$@")
    echo "$name: H100 $a | L40S $b  ($t, ${dep:-no-dep})" >&2; echo "$a:$b"
}

cpu_job () {             # NAME TIME MEM DEP CMD
    local name=$1 t=$2 mem=$3 dep=$4 cmd=$5
    if queued "$name"; then echo "SKIP $name (queued/running)" >&2; ids "$name"; return; fi
    local d=(); [[ -n $dep ]] && d=(--dependency=$dep)
    if [[ "${DRY:-0}" == 1 ]]; then echo "DRY  $name  CPU $t  ${dep:-no-dep}  :: $cmd" >&2; echo 0; return; fi
    local j; j=$(sbatch --parsable --account=gts-gchou3 --partition=cpu-small,cpu-medium --cpus-per-task=16 --mem=$mem \
        --time=$t --nice=$NICE --job-name=$name "${d[@]}" --output="$LOG/%x_%j.out" --error="$LOG/%x_%j.err" \
        --wrap="set -e; module load anaconda3; source activate $ENV; $cmd")
    echo "$name: CPU $j  ($t, ${dep:-no-dep})" >&2; echo "$j"
}

need_complete () {       # RUN N : refuse to plan evals on an incomplete dataset/view
    local n1 n2
    n1=$(ls "$ROOT/$1/dataset" 2>/dev/null | grep -c '\.pkl$' || true)
    n2=$(ls "$ROOT/$1/annos" 2>/dev/null | grep -c '_a\.pkl$' || true)
    [[ $n1 -eq $2 && $n2 -eq $2 ]] || { echo "ERROR: $1 has $n1 dataset / $n2 annos files, expected $2" >&2; exit 1; }
}

gen_dep_of () {          # RUN N -> "" if complete, "afterok:<id>" if its generation job is queued/running, else exit
    local n1 n2 j
    n1=$(ls "$ROOT/$1/dataset" 2>/dev/null | grep -c '\.pkl$' || true)
    n2=$(ls "$ROOT/$1/annos" 2>/dev/null | grep -c '_a\.pkl$' || true)
    [[ $n1 -eq $2 && $n2 -eq $2 ]] && { echo ""; return; }
    j=$(squeue -h -u "$USER" -n "cvpr_gen_$1" -o %i | head -1)
    [[ -n $j ]] || { echo "ERROR: $1 has $n1/$2 segments and no cvpr_gen_$1 job queued" >&2; exit 1; }
    echo "afterok:$j"
}
