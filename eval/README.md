# Evaluating detectors on ATLAS

This is the harness that turns an ATLAS attack into a number: run a detector on the
clean sequence, run it again on the attacked sequence, and report attack success rate
against clean mAP/NDS.

Every file here was copied byte-for-byte out of `Polar-Hierarchical-Mamba`'s
repository root. Hashes are in `atlas/PROVENANCE.json` under `mirrors`;
`tools/check_provenance.py` verifies them and `--vs-phm` catches upstream drift.

## The boundary, and why this directory needs a checkout of the pipeline repo

ATLAS lives here. The detectors it attacks do not — they are OpenPCDet models, and the
pipeline repo is a fork of OpenPCDet carrying PHiM, LOT, CUDA extensions and the
trained checkpoints. So the dependency runs one way:

    atlas-lidar/eval  ->  needs  ->  Polar-Hierarchical-Mamba (pcdet, checkpoints)
    atlas-lidar/atlas <-  needs  -   nothing but numpy

Nothing in `atlas/` imports torch at module scope, which is what keeps the fidelity
check runnable in CI. The scripts here import `pcdet.config`, `pcdet.models`,
`pcdet.datasets` and `pcdet.ops.iou3d_nms` at module scope and will not import at all
without that environment. That is expected, not a packaging bug.

```bash
pip install -e '/path/to/atlas-lidar[eval]'   # numpy, torch, scipy, easydict, pyyaml
cd /path/to/Polar-Hierarchical-Mamba && python setup.py develop   # provides pcdet
```

`atlas_generation` here is a symlink to `../atlas`. The copied scripts do
`sys.path.insert(0, dirname(__file__) + "/atlas_generation")` to reach
`nuscenes_atlas_common`, and the symlink is what lets them stay verbatim instead of
being edited on the way in.

## One required patch to pcdet

`pcdet-patches/waymo_dataset-atlas-asr.patch` is ATLAS evaluation logic that lives
*inside* the detector: `get_iou_threshold`, `find_best_match`, and the `spoof_gt`
branch that `WaymoDataset.evaluation()` calls to score attack success per frame.

It stays in the pipeline repo because it is called from a `pcdet` dataset class and
cannot run anywhere else. It is recorded here because it is ATLAS's code, and without
it the Waymo ASR numbers cannot be reproduced from a clean OpenPCDet. It applies
cleanly to the pipeline repo's `public_release` branch, and is already present on its
`ATLAS` branch.

The rest of that repo's `pcdet/` changes are PHiM/LOT model work and stay there.

## What runs what

| | |
|---|---|
| `point_injection.py` | Waymo point injection against a detector |
| `point_injection_nuscenes.py` | nuScenes point injection |
| `eval_matrix_kitti.py`, `eval_matrix_nuscenes.py` | the full detector × attack matrix |
| `atlas_injection_eval_only.py` | scores cached predictions; numpy-only apart from a lazy `iou3d_nms` import |
| `cache_injection_preds.py` | caches detector predictions so the matrix can be re-scored without re-running inference |
| `ablate_intensity_nuscenes.py` | intensity-calibration ablation |
| `analyze_schedule_variants.py` | attack-schedule comparison, Wilcoxon signed-rank; needs only numpy + scipy |
| `make_qualitative_figs.py` | the BEV figures in `results/qualitative_figs/` |
| `run_*.sh` | the drivers that were actually used for the reported runs |
| `cfgs/` | the three dataset configs added for ATLAS generation, from the pipeline repo's `tools/cfgs/dataset_configs/` |

`cache_injection_preds.py` also inserts `<repo>/tools` on `sys.path`, which resolved
to the pipeline repo's `tools/` (train/test utilities) and does **not** resolve here.
Run it from the pipeline repo, or point that path at the checkout.

## Interchange with the sandbox

The sandbox exports `.atlasattack` bundles; `tools/import_attack.py` converts them
into the per-frame pickles these scripts consume. That is the supported round trip
between a hand-built attack and a real detector score.
