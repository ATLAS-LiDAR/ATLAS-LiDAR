#!/bin/bash
source /home/mellonzhang/miniconda3/etc/profile.d/conda.sh
conda activate atlas
cd /home/mellonzhang/rsch/Polar-Hierarchical-Mamba/tools
export CUDA_HOME=/home/mellonzhang/miniconda3/envs/atlas
export PATH=$CUDA_HOME/bin:$PATH
A=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/nuscenes_trainval/atlas_val
CKPT=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/nuscenes/checkpoints
python ../ablate_intensity_nuscenes.py \
  --atlas_dir "$A" --ckpt_dir "$CKPT" \
  --out_csv "$A/atlas_intensity_sweep.csv" \
  --work_dir "$A/ablate_sweep_work" \
  --models pp_multihead \
  --splits inject_relative_easy \
  --intensities -1 5 15 30 60 100 150 200 255
echo "SWEEP EXIT=$?"
