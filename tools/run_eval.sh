#!/usr/bin/env bash
# Score sandbox-generated attacks with the real detectors in
# Polar-Hierarchical-Mamba.
#
#   tools/run_eval.sh /tmp/asr/'*.atlasattack' [model]
#
# WHY THE PYTHONPATH LINE EXISTS
# ------------------------------
# The `atlas` conda env is internally consistent (torch 2.11.0+cu128,
# torchvision 0.26.0+cu128, nvidia-nccl-cu12 2.28.9) but ~/.local/lib/python3.10
# /site-packages *shadows* it with an older set left over from a torch 2.7
# install: torchvision 0.22.0+cu128 and nvidia-nccl-cu12 2.26.2.
#
# User-site precedes env site-packages on sys.path, which broke two things:
#
#   1. torch preloads its CUDA libs by looking the packages up through importlib
#      metadata, so it found ~/.local's nccl 2.26.2 and loaded it RTLD_GLOBAL.
#      That build has no ncclCommShrink, so libtorch_cuda.so then failed with
#      "undefined symbol: ncclCommShrink" - which reads like a torch/NCCL version
#      mismatch but is really a shadowing problem.
#   2. pcdet imports torchvision, got ~/.local's 0.22 (built for torch 2.7), and
#      died with "operator torchvision::nms does not exist".
#
# Putting the env's own site-packages first on PYTHONPATH fixes both, changes no
# packages, and is entirely reversible. Note PYTHONNOUSERSITE=1 does NOT work:
# the env is missing typing_extensions and relies on ~/.local for it.
#
# The permanent fix, if you want one, is to remove the stale torch-2.7-era
# packages from ~/.local (torchvision 0.22, nvidia-nccl-cu12 2.26.2) - but that
# is your call, since other envs may lean on them.
set -euo pipefail

# WHERE THE PIECES LIVE, AFTER THE SPLIT
# The eval scripts moved into this repo (eval/), so they are no longer at $PHM's root
# the way they were when this was written; $SANDBOX/eval holds them now.
#
# $PHM is NOT the submodule, and cannot be. cwd decides where pcdet resolves the KITTI
# dataset root, and a submodule only ever contains code - eval/phim/data/kitti has
# ImageSets and nothing else. Point this at a checkout with KITTI actually installed
# (calib, velodyne, kitti_infos_*.pkl), which for the pipeline repo means the NAS mount.
#
# Getting this wrong fails in the worst possible way: KittiDataset loads 0 samples,
# get_calib asserts on a missing file, and eval_matrix_kitti.py's `except Exception:
# continue` counts every frame as an injection miss - so you get a clean-looking
# ASR=0.0000 (0/N) instead of an error. Hence the preflight checks below.
SANDBOX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHM="${PHM_ROOT:-$HOME/rsch/Polar-Hierarchical-Mamba}"
CKPT="${CKPT_DIR:-$PHM/data/kitti/checkpoints}"
ENV_PY="${ATLAS_PY:-$HOME/miniconda3/envs/atlas/bin/python}"
SP="$("$ENV_PY" -c 'import site,sys; print(site.getsitepackages()[0])')"

die() { echo "run_eval: $1" >&2; exit 1; }
[[ -d "$PHM/tools/cfgs" ]]              || die "no $PHM/tools/cfgs - set PHM_ROOT to a Polar-Hierarchical-Mamba checkout"
[[ -d "$PHM/data/kitti/training/calib" ]] || die "no KITTI calib under $PHM/data/kitti - the dataset must be installed there, or ASR comes back a silent 0"
[[ -f "$PHM/data/kitti/kitti_infos_val.pkl" ]] || die "no kitti_infos_val.pkl under $PHM/data/kitti - run the pipeline repo's dataset preparation first"
[[ -d "$CKPT" ]]                        || die "no checkpoints at $CKPT - set CKPT_DIR"

BUNDLES="${1:?usage: run_eval.sh '<glob of .atlasattack>' [model]}"
MODEL="${2:-pointpillar}"
WORK="$(mktemp -d)"
mkdir -p "$WORK/inject_easy"

echo "== converting bundles -> per-frame pkl (KITTI layout)"
python3 "$SANDBOX/tools/import_attack.py" "$BUNDLES" \
  --segment sandbox --layout kitti \
  --out-dir "$WORK/inject_easy" --annos-dir "$WORK/annos" --verify

echo
echo "== scoring with the real $MODEL checkpoint"
# cwd must be $PHM/tools: eval_matrix_kitti.py resolves cfgs/ relative to it.
# The script itself now lives here, in this repo.
cd "$PHM/tools"
PYTHONPATH="$SP" "$ENV_PY" "$SANDBOX/eval/eval_matrix_kitti.py" \
  --atlas_dir "$WORK" \
  --ckpt_dir "$CKPT" \
  --out_csv "$WORK/asr.csv" \
  --work_dir "$WORK/logs" \
  --only_model "$MODEL" \
  --skip_clean 2>&1 | grep -E 'MODEL|ASR=|ERROR' || true

echo
echo "== results"
grep -v 'missing dir' "$WORK/asr.csv"
echo
echo "artifacts in $WORK"
