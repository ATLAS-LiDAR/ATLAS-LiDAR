# Run records, not portable scripts

These are the exact shell drivers used to produce the numbers in `results/`. They are
kept verbatim — hashes are in `atlas/PROVENANCE.json` — because "here is the command
that produced Table 1" is worth more than a tidied-up version that was never actually
run.

**They will not work unmodified.** Each one hardcodes the original machine:

```bash
source /home/mellonzhang/miniconda3/etc/profile.d/conda.sh   # a specific conda install
conda activate atlas                                          # a specific env name
cd /home/.../Polar-Hierarchical-Mamba/tools                   # the pipeline repo, pre-split
export CUDA_HOME=/home/mellonzhang/miniconda3/envs/atlas
ATLAS=/home/mellonzhang/mmt-nas-1/.../kitti/atlas_val          # a NAS mount
CKPT=/home/mellonzhang/mmt-nas-1/.../kitti/checkpoints
```

To reuse one, expect to change four things: the conda activation, the `cd` (now
`eval/phim/tools`), the generated-data directory, and the checkpoint directory. The
`python ../<script>.py` invocations refer to what is now `eval/`.

Nothing here is on a CI path, so nothing silently depends on those paths still existing.

| | |
|---|---|
| `run_kitti_eval.sh` | the KITTI detector × attack matrix |
| `run_kitti_pointrcnn.sh` | the same, PointRCNN only |
| `run_sched_eval.sh` | nuScenes across attack schedules |
| `run_ablate_intensity.sh` | intensity-calibration ablation |
| `run_ablate_sweep.sh` | the same, swept over parameter values |
