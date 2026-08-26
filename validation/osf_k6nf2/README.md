# Real-world validation against the OSF `k6nf2` capture

Analysis code for comparing ATLAS-generated adversarial LiDAR against a **real
physical attack capture**: the Velodyne VLP-16 packet traces from

> Cao et al., "You Can't See Me: Physical Removal Attacks on LiDAR-based
> Autonomous Vehicles Driving Frameworks", USENIX Security 2023
> (arXiv:2210.09482), OSF archive `k6nf2`.

The raw `.pcap` files (~440 MB) are **not** in git — download the OSF archive and
unzip it into this directory so that `Multimode Analysis/{last,dual}_mode.pcap`
exists.

## What the capture actually is

A **removal** attack (PRA), not an injection attack. Saturated echoes are
injected *below* the sensor's Minimum Operational Threshold; because the sensor
prioritises the strongest echo and the firmware discards sub-MOT returns, the
genuine object's returns are discarded. Removal is achieved *via* injection.

Two consequences that took a while to pin down:

* The pcaps are recorded **pre-MOT-filter**, so removal does NOT appear as
  missing returns. It appears as genuine echoes being **range-replaced** by the
  spoof echo on the same ray. Occupancy-depletion analysis finds nothing.
* Removal is strongly **peaked in azimuth**. Averaging over the whole saturated
  envelope (~6.8 deg) mixes the aim point with the off-target fringe and
  understates the rate by roughly 2x. Compare `p_remove` against the **aim
  point**, not the envelope average.

## Scripts

| script | what it does |
|---|---|
| `parse_velo.py` | dependency-free pcap + Velodyne UDP decoder (no dpkt/scapy) |
| `real_removal_v3.py` | measures the real removal rate, wedge width, channel span |
| `compare_removal.py` | like-for-like: real vs the ATLAS `removal_az*` splits |
| `pc_stats.py` | sensor-physics invariants (I1-I6) vs real hardware, with a clean-data control |
| `extract_spoof_real.py` | isolates spoofed returns (reflectivity >= 200 + beyond background) |
| `track_target.py` | per-frame bearing/range of the spoofed cluster |
| `cell_occupancy.py`, `ppc_control.py` | points-per-sampling-cell, with a real-object control |
| `compare_real_vs_atlas.py`, `realizability.py` | angular footprint / density comparisons |
| `find_spoof.py`, `scan_refl.py`, `scan_far.py`, `sector_detail.py`, `fill_generated.py` | exploratory helpers |

## Reproducing the headline numbers

```bash
conda activate atlas

# real removal characteristics (0.4 deg binning is correct for the VLP-16 --
# the two interleaved firings share a cell in this decoder)
python real_removal_v3.py

# real vs ATLAS removal splits
python compare_removal.py \
    --nuscenes_dir <nuscenes_trainval>/atlas_val

# sensor-physics invariants. Use KITTI: it is single-sweep, so per-ray
# occupancy is comparable to a real revolution. nuScenes is 10-sweep
# accumulated and fails I4 by construction (~9 returns/cell).
python pc_stats.py --dataset kitti \
    --atlas_dir <kitti>/atlas_val \
    --kitti_root <kitti> \
    --splits removal_az20 inject_easy inject_hard
```

## Results summary

**Removal strength** (fraction of genuine returns destroyed in the wedge):

| | rate |
|---|---|
| real, aim point / best-aimed frames | 0.83-0.87 |
| Cao et al. reported (moving car) | 0.836-0.927 |
| ATLAS `removal_az*` | 0.90-1.00 |

ATLAS is 1.0-1.2x the demonstrated rate. Vertical span agrees (16/16 real,
20-32/32 ATLAS = full VFOV both sides). `az50`/`az60` exceed the paper's 45 deg
max demonstrated attack angle and are flagged `EXTRAP`.

**Sensor-physics invariants** (KITTI; `clean` = unmodified KITTI val bins, the
control that separates generator error from dataset error):

| invariant | real VLP-16 | ATLAS | clean KITTI |
|---|---|---|---|
| I6 ring-arc gap / az_res | 1.000 | 1.039 | 1.039 |
| I4 single-return-per-ray | 0.978 | 0.881 | 0.830 |
| per-ring density (pts/ring/frame) | - | 352-433 | 411 |
| I1 2 mm range grid | 1.000 | 0.101 | 0.105 |
| I3 azimuth grid | 0.119 | 0.100 | 0.101 |

ATLAS matches the `r*dtheta` arc-spacing law to 3.9% and beats clean KITTI on
per-ray occupancy. I1/I3 fail identically for ATLAS and clean KITTI, i.e. they
are properties of the released dataset (rectified float coords), not generation
artifacts.
