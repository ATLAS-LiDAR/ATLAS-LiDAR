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

WHAT IS NOT HERE
Generating the full attacked datasets and scoring them against real detectors
needs OpenPCDet, torch, trained checkpoints and the raw KITTI/nuScenes archives.
That half stays in the paper's pipeline repo; this package is the part that has no
such dependencies. The `.atlasattack` bundle written by the sandbox is the
interchange format between the two.

CALIBRATION ASSETS
The modules load beam inclinations, azimuth resolution and range-conditioned
intensity statistics from `.npy` files next to themselves. Two nuScenes sets ship
here and they are NOT interchangeable - see PROVENANCE.md. Override with
NUSCENES_ATLAS_ASSET_DIR / KITTI_ATLAS_ASSET_DIR.

Submodules are deliberately not imported here: removal.py pulls in matplotlib and
plotly for its plotting helpers, and `import atlas` should not require a plotting
stack.
"""

__version__ = "0.1.0"
__all__ = ["removal", "kitti_atlas_common", "nuscenes_atlas_common"]
