# Detector patches

The nuScenes detectors evaluated by `eval/harness/` are third-party repositories. These
patches hold everything that had to change in them, so each can be reproduced from a
clean clone of the upstream commit it was tested against:

| repository | upstream commit | patch | what it does |
|---|---|---|---|
| [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | `233f849` | `openpcdet/atlas-nuscenes.patch` | ATLAS hook |
| [LION](https://github.com/happinesslz/LION) | `f7cbdf9` | `lion/atlas-nuscenes.patch` | ATLAS hook, no-GT-sampling config |
| [DSVT](https://github.com/Haiyang-W/DSVT) | `8cfc2a6` | `dsvt/atlas-nuscenes.patch` | ATLAS hook, no-GT-sampling config |
| [CenterPoint](https://github.com/tianweiy/CenterPoint) | `3cf7d87` | `centerpoint/compat.patch` | runs on current Python / NumPy / Numba / PyTorch |
| | | `centerpoint/no-gt-config.patch` | no-GT-sampling config |
| [CenterFormer](https://github.com/TuSimple/centerformer) | `5a949b8` | `centerformer/compat.patch` | runs on current Python / spconv 2.x / PyTorch |
| | | `centerformer/no-gt-config.patch` | no-GT-sampling config |

Apply from the repository root:

```bash
git -C /path/to/LION apply /path/to/ATLAS-LiDAR/eval/detector-patches/lion/atlas-nuscenes.patch
```

## The ATLAS hook (OpenPCDet, LION, DSVT)

The patch adds `pcdet/datasets/nuscenes/atlas_eval.py` and `numpy_compat.py`, and
touches three files:

- `NuScenesDataset` restricts itself to the attacked frames when the config carries
  `ATLAS_MANIFEST`, and loads each frame's attacked cloud instead of the clean sweeps;
- `tools/eval_utils/eval_utils.py` scores ASR instead of full-split metrics when
  `--atlas_attack` is set (and `--predictions_only` stops after `result.pkl`);
- `tools/test.py` gains `--atlas_attack`, `--atlas_data`, `--atlas_config`,
  `--atlas_cache`, `--atlas_target_class`, `--atlas_iou_thresh`,
  `--atlas_score_thresh`.

Without `--atlas_*` flags the repository behaves exactly as before. The hook also works
on its own, without the harness:

```bash
cd LION/tools
python test.py --cfg_file cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
    --ckpt /path/to/checkpoint_epoch_36.pth --batch_size 4 \
    --atlas_attack inject --atlas_data /data/ATLAS-nuScenes --atlas_config inject_global_easy
```

`atlas_eval.py` is byte-identical to `eval/harness/nuscenes/atlas_eval.py`;
`eval/harness/tests/test_nuscenes.py` checks that every patch still ships the same file.
`numpy_compat.py` lets the devkit's detection metrics run under NumPy ≥ 1.24, where
`np.float` no longer exists.

## No-GT-sampling configs

Database GT sampling ("copy-paste" augmentation) pastes real objects from other scenes
into each training cloud. Because ATLAS injection is itself a paste of real object
returns, the evaluation compares each detector trained with and without it. Each
`*_no_gt` config is its upstream counterpart with only the `gt_sampling` augmentation
removed; everything else, including the schedule, is unchanged:

| | with GT sampling (upstream) | without |
|---|---|---|
| LION | `lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml` | `..._no_gt.yaml` |
| DSVT | `dsvt_plain_1f_onestage_nusences.yaml` | `..._no_gt.yaml` |
| CenterPoint | `nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py` | `..._no_gt.py` |
| CenterFormer | `nuscenes_centerformer_separate_detection_head.py` | `..._no_gt.py` |

## Compatibility fixes (CenterPoint, CenterFormer)

These code bases predate current Python and their dependencies. The fixes change no
model behaviour:

- `collections.Iterable` / `Sequence` / `Mapping` → `collections.abc` (Python ≥ 3.10);
- CenterPoint `geometry.py`: Numba nopython mode no longer accepts a Python list as an
  array index; the same edge vectors are computed with scalar indices;
- CenterFormer `scn.py`: spconv 2.x sparse tensors are immutable, so features are set
  with `replace_feature`;
- `tools/train.py`: accept `--local-rank` and read `LOCAL_RANK`, as `torchrun` passes it;
- CenterPoint `train.py`: fall back to PyTorch's native `SyncBatchNorm` when Apex is not
  installed.

The harness runs the det3d models' own input pipelines and only replaces the
point-cloud loader, so these repositories need no ATLAS hook.
