#!/bin/bash
source /home/mellonzhang/miniconda3/etc/profile.d/conda.sh
conda activate atlas
cd /home/mellonzhang/rsch/Polar-Hierarchical-Mamba/tools
export CUDA_HOME=/home/mellonzhang/miniconda3/envs/atlas
export PATH=$CUDA_HOME/bin:$PATH
ATLAS=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/kitti/atlas_val
CKPT=/home/mellonzhang/mmt-nas-1/mellon/data/datasets/kitti/checkpoints
python ../eval_matrix_kitti.py \
  --atlas_dir "$ATLAS" \
  --ckpt_dir "$CKPT" \
  --out_csv "$ATLAS/atlas_kitti_results.csv" \
  --work_dir "$ATLAS/eval_work"
echo "KITTI EVAL EXIT=$?"
