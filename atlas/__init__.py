"""ATLAS: the reference implementation of the LiDAR spoofing attack.

This is the attack itself - the physics and the algorithms - extracted from the
paper's original codebase so it can be used, tested and cited without dragging a
detector, a CUDA build and a checkpoint directory along with it.

    from atlas import removal, kitti_atlas_common, nuscenes_atlas_common

Three modules, ~1,400 lines, numpy only:

    removal              A-HFR sector removal: aim at a tracked target, delete
                         the returns inside an azimuth x elevation wedge.
    kitti_atlas_common   KITTI injection: place a real vehicle's returns into a
                         frame, resolving occlusion on the scan lattice.
    nuscenes_atlas_common  the same for nuScenes, plus sequence-level machinery
                         (ego poses, attack schedules, global vs relative placement).

ALSO IN THIS DIRECTORY, BUT NOT IN THIS PACKAGE
`generate_data_*.py`, `build_*_assets.py`, `helpers_ptt.py` and `export_*_hf.py` are
the dataset generation layer, mirrored from the pipeline repo alongside the three
modules above. They are flat here rather than in a subdirectory because they use
sibling imports (`import nuscenes_atlas_common as C`), which is what lets them stay
byte-identical to the originals. They need torch and the plotting stack
(`pip install -e '.[gen]'`) and import `pcdet` lazily inside their data loaders.

Scoring attacks against real detectors lives in `../eval/`, which additionally needs
an OpenPCDet checkout and trained checkpoints.

WHAT IS NOT HERE
The detectors. This package is the attack, and it has no such dependencies.
The `.atlasattack` bundle written by the sandbox is the interchange format.

CALIBRATION ASSETS
The modules load beam inclinations, azimuth resolution and range-conditioned
intensity statistics from `.npy` files next to themselves. Two nuScenes sets ship
here and they are NOT interchangeable - see PROVENANCE.md. Override with
NUSCENES_ATLAS_ASSET_DIR / KITTI_ATLAS_ASSET_DIR.

Submodules are deliberately not imported here, so that `import atlas` costs nothing
and cannot fail. The three modules above are numpy-only, but their neighbours in this
directory are not - helpers_ptt.py imports torch, matplotlib and plotly at module
scope - and an eager `__init__` would drag that in.
"""

__version__ = "0.1.0"
__all__ = ["removal", "kitti_atlas_common", "nuscenes_atlas_common"]
