# Evaluating detectors on ATLAS

This is the harness that turns an ATLAS attack into a number: run a detector on the
clean sequence, run it again on the attacked sequence, and report attack success rate
against clean mAP/NDS.

Every file here was copied byte-for-byte out of `Polar-Hierarchical-Mamba`'s
repository root. Hashes are in `atlas/PROVENANCE.json` under `mirrors`;
`tools/check_provenance.py` verifies them and `--vs-phm` catches upstream drift.

## The boundary, and the submodule that spans it

ATLAS lives here. The detectors it attacks do not — they are OpenPCDet models, and the
pipeline repo is a fork of OpenPCDet carrying PHiM, LOT, CUDA extensions and the
trained checkpoints. So the dependency runs one way:

    atlas-lidar/eval  ->  needs  ->  Polar-Hierarchical-Mamba (pcdet, checkpoints)
    atlas-lidar/atlas <-  needs  -   nothing but numpy

Nothing in `atlas/` imports torch at module scope, which is what keeps the fidelity
check runnable in CI. The scripts here import `pcdet.config`, `pcdet.models`,
`pcdet.datasets` and `pcdet.ops.iou3d_nms` at module scope and will not import at all
without that environment. That is expected, not a packaging bug.

`phim/` is that repo, vendored as a **git submodule pinned to its `ATLAS` branch** —
not `public_release`, because the ATLAS branch is the one carrying the evaluation hook
described below and the nuScenes/KITTI configs these runs used.

```bash
git submodule update --init eval/phim          # or clone with --recurse-submodules
pip install -e '.[eval]'                       # numpy, torch, scipy, easydict, pyyaml
cd eval/phim && python setup.py develop        # builds the CUDA extensions, provides pcdet
```

The last step needs a CUDA toolchain and takes a while; it is the OpenPCDet build, not
anything ATLAS adds. Checkpoints are not in the submodule — get them from the pipeline
repo's own instructions.

`atlas_generation` here is a symlink to `../atlas`. The submodule no longer has a copy —
ATLAS was removed from the pipeline repo once it lived here, so `phim/` is purely the
perception models plus the one evaluation hook below.

The ATLAS code is still in that repo's *history*, which is how
`tools/check_provenance.py --vs-phm` verifies this extraction: it reads blobs out of
commit `a11dc37`, the last one containing them.

Why the symlink: the copied scripts do
`sys.path.insert(0, dirname(__file__) + "/atlas_generation")` to reach
`nuscenes_atlas_common`, and the symlink is what lets them stay verbatim instead of
being edited on the way in.

## One required patch to pcdet

`pcdet-patches/waymo_dataset-atlas-asr.patch` is ATLAS evaluation logic that lives
*inside* the detector: `get_iou_threshold`, `find_best_match`, and the `spoof_gt`
branch that `WaymoDataset.evaluation()` calls to score attack success per frame.

It stays in the pipeline repo because it is called from a `pcdet` dataset class and
cannot run anywhere else. It is recorded here because it is ATLAS's code, and without
it the Waymo ASR numbers cannot be reproduced from a clean OpenPCDet.

**You do not need to apply it** — it is already present in `phim/`, which is pinned to
the `ATLAS` branch. It is kept here so the logic is versioned on the attack's side, and
so it can be reapplied to a different OpenPCDet fork; it applies cleanly to the pipeline
repo's `public_release` branch.

It is also the one place where a change upstream silently changes results here. The rest
of that repo's `pcdet/` changes are PHiM/LOT model work and stay there.

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
| `make_qualitative_figs.py` | the BEV figures for the paper (written wherever you point `--out_dir`) |
| `scripts/` | the drivers actually used for the reported runs, kept verbatim — see [scripts/README.md](scripts/README.md) |
| `cfgs/` | the three dataset configs added for ATLAS generation, from the pipeline repo's `tools/cfgs/dataset_configs/` |

`cache_injection_preds.py` also inserts `<repo>/tools` on `sys.path`, which resolved
to the pipeline repo's `tools/` (train/test utilities) and does **not** resolve here.
Run it from the pipeline repo, or point that path at the checkout.

## Interchange with the sandbox

The sandbox exports `.atlasattack` bundles; `tools/import_attack.py` converts them
into the per-frame pickles these scripts consume. That is the supported round trip
between a hand-built attack and a real detector score.
