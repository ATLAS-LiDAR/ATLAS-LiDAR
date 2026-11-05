# Multi-detector evaluation harness

`eval/point_injection*.py` and `eval/eval_matrix_*.py` score ATLAS against OpenPCDet
models one run at a time. This harness scores many detectors from different code bases
against every ATLAS configuration in one invocation, and reports accuracy on the
attacked frames next to the attack success rate.

| | |
|---|---|
| `nuscenes/run_atlas_eval.py` | nuScenes: ASR, plus official AP / mAP / NDS **on the attacked frames only**; `clean` runs the full val split |
| `nuscenes/atlas_eval.py` | reads ATLAS-nuScenes Parquet, scores ASR; the same file is installed into detector repos by `../detector-patches/` |
| `nuscenes/attacked_metrics.py` | the devkit's `NuScenesEval`, restricted to an exact set of sample tokens |
| `nuscenes/add_class_recall.py` | adds real-object recall for one class (e.g. pedestrians near a phantom pedestrian) to a summary |
| `waymo/run_atlas_eval.py` | Waymo: ASR plus the repository's standard Waymo metrics |
| `waymo/compute_ap.py` | Waymo L1/L2 AP/APH from saved `result.pkl` files, without re-running inference |
| `common.py` | GPU pool, manifests, resumable receipts, summaries |
| `tests/` | `python -m unittest discover -s eval/harness/tests` — no GPU, datasets or detectors needed |

## How a run is organised

Every (model, configuration) pair is an independent job in its own single-GPU process,
so detectors with incompatible dependencies run side by side. Give each model its own
interpreter (`python`) or `pythonpath` in the manifest when they need separate
environments.

- **Scheduling.** `--gpus` (default: `CUDA_VISIBLE_DEVICES`) lists the devices;
  `--slots_per_gpu 2` runs two jobs per device so one job's CPU-bound metric phase
  overlaps another's inference. A job that dies with an uncorrectable ECC error takes
  its GPU out of the pool; the remaining jobs continue on the other devices.
- **Resuming.** Each job writes `done.json` with a signature of its inputs (checkpoint,
  config, the harness code and the detector's `models/` and `datasets/` sources, data
  shards, thresholds). `--resume` skips a job only when the signature is unchanged, and
  refuses to mix results when it is not. A failed retry never replaces a successful
  row in the summary.
- **Output.** `<output_dir>/summary.{csv,json}` has one row per job;
  `<output_dir>/<model>/<configuration>/` holds `worker.log`, `result.pkl`,
  `atlas_asr.json` (per-frame records) and the devkit metric files.

## Manifests

```json
{
  "lion_no_gt": {
    "backend": "pcdet", "repo": "${LION_ROOT}",
    "cfg": "tools/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim_no_gt.yaml",
    "ckpt": "${CKPT_ROOT}/lion_no_gt/checkpoint_epoch_{epoch}.pth", "epochs": [9, 18, 27, 36]
  }
}
```

`${VAR}` and `~` are expanded. A relative `repo` or `ckpt` is relative to the manifest
file; a relative `cfg` is relative to `repo`. A `{epoch}` checkpoint with an `epochs`
list expands to one model per epoch (`lion_no_gt_ep9`, ...), which is how robustness
over the course of training is measured. Optional keys: `python`, `pythonpath`, and
`variant` (`centerformer` copies `test_cfg` thresholds into the neck, as that code base
expects). See `nuscenes/manifest.example.json` and `waymo/manifest.example.json`.

## nuScenes

```bash
python eval/harness/nuscenes/run_atlas_eval.py \
    --manifest models.json \
    --atlas_data /data/ATLAS-nuScenes \
    --dataroot /data/nuscenes/v1.0-trainval \
    --output_dir eval/output/nuscenes \
    --gpus 0,1,2,3 --slots_per_gpu 2 --resume
```

`--dataroot` holds `v1.0-trainval/` (devkit metadata) and
`nuscenes_infos_10sweeps_val.pkl`. `--configs` defaults to every configuration under
`<atlas_data>/data/`; add `clean` for the full validation split. Use `--dry_run` first:
it checks every path and lists the jobs without importing torch.

**Backends.**
- `det3d` (CenterPoint, CenterFormer) runs the repository unmodified apart from the
  compatibility fixes in `../detector-patches/`; the harness replaces the first stage
  of its input pipeline with a loader for the attacked clouds. Converting the
  predictions to devkit format uses OpenPCDet's `nuscenes_utils`, taken from
  `--pcdet_repo` (default `eval/phim`).
- `pcdet` (OpenPCDet, LION, DSVT, ...) needs the ATLAS hook applied — see
  `../detector-patches/README.md`.

Both load checkpoints with `strict=True`: a weight that does not fit the model fails the
job instead of producing a silently degraded score.

**What is measured.**
- Only the attacked frames shipped in ATLAS-nuScenes are evaluated, each with its own
  pre-accumulated 10-sweep cloud (columns `x, y, z, intensity, time`); no clean sweeps
  are added.
- **ASR**: for `inject_*`, the fraction of attacked frames where a prediction of the
  target class overlaps the phantom box `spoof_gt` at rotated 3D IoU ≥ threshold; for
  `removal_*`, the fraction where the erased object is no longer detected. The
  denominator is every attacked frame, not conditioned on the clean model detecting
  anything.
- **Target class and IoU** follow from the configuration name: car configurations
  (`inject_global_easy`, `removal_az20`) score cars at IoU 0.3;
  `inject_*_pedestrian_*` scores pedestrians and `inject_*_cyclist_*` scores the
  nuScenes `bicycle` class, both at 0.5. `--target_class` / `--iou_thresh` override.
- **AP / mAP / NDS** are the devkit's own metrics over exactly the attacked tokens.
  Every real object in those frames is a ground truth; the injected phantom is not.
  Frames with no predictions count as empty samples, not as missing ones. These are
  subset metrics and are not comparable to full-val leaderboard numbers; compare them
  with the `clean` row of the same model, or with the same subset of a clean run.

For phantom-pedestrian runs, `add_class_recall.py <output_dir> --class pedestrian` adds
the recall of real pedestrians in the attacked frames.

## Waymo

```bash
python eval/harness/waymo/run_atlas_eval.py \
    --manifest models.json --data_root /data/waymo_atlas \
    --output_dir eval/output/waymo --gpus 0,1,2,3 --resume
```

Each configuration is a directory under `--data_root` in the OpenPCDet Waymo layout
whose infos carry `spoof_gt` (`ImageSets/val.txt`, `waymo_processed_data_v0_5_0/`).
The model repository defaults to `eval/phim`; a manifest entry can name another. The
repository's `eval_one_epoch` produces `result.pkl` and the Waymo metrics unchanged.
ASR is then scored on every frame with a valid `spoof_gt`, at IoU 0.7 for Vehicle and
0.5 for Pedestrian/Cyclist, matching `../pcdet-patches/waymo_dataset-atlas-asr.patch`.
The summary adds per-class L1/L2 AP/APH and their means.

`compute_ap.py --output_dir <out> --data_root <data>` computes the same Waymo AP/APH
from existing `result.pkl` files, for runs that stopped at ASR.
