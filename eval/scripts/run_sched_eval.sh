#!/bin/bash
# Evaluate the nuScenes checkpoints on the 24 SCHEDULE variants in atlas_val_sched/.
#
# Runs the two continuity regimes as two passes over the 12 base split names,
# selected with --split_suffix. Clean mAP/NDS is skipped: it is unaffected by the
# attack schedule and is already recorded in atlas_nuscenes_results.csv.
#
# --per_scene_dir is required here: with variable attack lengths a frame-pooled
# ASR weights long attacks far more than short ones, so sequence-level and
# macro-averaged ASR (computed by analyze_schedule_variants.py from these dumps)
# are the fair summaries.
source /home/mellonzhang/miniconda3/etc/profile.d/conda.sh
conda activate atlas
cd /home/mellonzhang/rsch/Polar-Hierarchical-Mamba/tools
export CUDA_HOME=/home/mellonzhang/miniconda3/envs/atlas
export PATH=$CUDA_HOME/bin:$PATH

ATLAS=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/nuscenes_trainval/atlas_val_sched
CKPT=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/nuscenes/checkpoints
OUTCSV=$ATLAS/atlas_nuscenes_sched_results.csv

for sfx in _cont _intermit; do
  echo "===== ASR pass for suffix ${sfx} $(date +%H:%M:%S) ====="
  python ../eval_matrix_nuscenes.py \
    --atlas_dir "$ATLAS" \
    --ckpt_dir "$CKPT" \
    --out_csv "$OUTCSV" \
    --work_dir "$ATLAS/eval_work" \
    --per_scene_dir "$ATLAS/per_scene" \
    --split_suffix "$sfx" \
    --skip_clean
  echo "pass ${sfx} EXIT=$?"
done
echo "===== SCHED EVAL DONE $(date +%H:%M:%S) ====="
