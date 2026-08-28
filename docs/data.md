# Data and sources

## Data

`tools/export_frames.py` reads KITTI `velodyne` bins and nuScenes `LIDAR_TOP`
sweeps with their lidar-frame annotations and writes `.atlasframe` (int16
positions at 5 mm, uint8 intensity — ~820 kB for a 120k-point HDL-64E frame).

**Single sweeps only.** nuScenes infos accumulate 10 sweeps, which puts ~9
returns in every cell and destroys the one-return-per-ray property the whole
cell-occupancy model rests on.

```sh
python3 tools/export_frames.py --list kitti      # browse candidates
python3 tools/export_frames.py --library         # export the built-in set
```

Users can also drop in raw frames with no conversion: KITTI `.bin` (stride 4) and
nuScenes `.pcd.bin` (stride 5) are parsed directly in the browser, with the
stride inferred and flagged when genuinely ambiguous.

## Sources

- Cao et al., *You Can't See Me: Physical Removal Attacks on LiDAR-based
  Autonomous Vehicles Driving Frameworks*, USENIX Security 2023
  (arXiv:2210.09482) — the real capture all calibration anchors come from.
- `atlas/nuscenes_atlas_common.py` and `atlas/removal.py` (in this repo; see
  [atlas/PROVENANCE.md](../atlas/PROVENANCE.md)). For the evaluation half:
  `Polar-Hierarchical-Mamba/ATLAS_REAL_WORLD_VALIDATION.md` and
  `osf_k6nf2/realizability.py` in the pipeline repo.
