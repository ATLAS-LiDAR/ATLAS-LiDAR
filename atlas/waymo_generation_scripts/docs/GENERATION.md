# Generation internals (how every dataset is built)

## Data
- Waymo Open Dataset v1.x processed by OpenPCDet (`waymo_processed_data_v0_5_0`), TOP LiDAR + others as pcdet
  loads them; features x, y, z, intensity, elongation (pcdet applies **tanh** to intensity and elongation), plus a
  lag/timestamp column in stored frames. Point range x, y in [-75.2, 75.2] m, z in [-2, 4] m.
- 195 validation segments are common to the single-frame and 4-frame loaders; the study uses 50 of them
  (`data/segments_50_seed42.txt`), drawn by `np.random.default_rng(42)` over the sorted common list.

## Schedule
Each segment: alternating 32-frame blocks starting clean (0-31 clean, 32-63 attack, 64-95 clean, ...). Injection
uses the whole attack block (`--continuous_min_len 32 --continuous_max_len 32`). Flicker (`--random p`): every
attack frame fires independently with probability p (0.5); ASR is scored on fired frames.

## Point injection (generators/gen_atlas_{global,relative}_cvpr.py)
1. Per attack run, `run_seed = 42 + global frame index of the run's first frame`; draw a trace (uniform over the
   library), offset ahead U(15, 20) m, lateral U(-3, 3) m, yaw U(-0.2, 0.2) rad. The trace is centred and rotated
   to yaw 0 first.
2. Global: placement in the ego frame of the run's LAST frame, converted to world; every frame transforms the
   world-fixed object into its own ego frame. Relative: the same ego-relative placement on every frame.
3. Ground anchoring each frame: scene points within 2 m (xy) of the spoof centre; if > 20, shift the spoof so its
   lowest point sits at their 5th-percentile z.
4. Jitter (only `*_jpt_*` / `*_jfr_*`): each spoof point moved along its own ray by N(0, sigma_inner) and the
   whole object by one shared N(0, sigma_inter) per frame; the stored `spoof_gt` is shifted with the shared
   offset (realized box), the aimed box is kept as `spoof_gt_intended`. Seeded by hash(seed:segment:frame).
5. Raycasting: spherical coordinates, 0.1358 deg azimuth bins x 64 TOP-LiDAR beam inclinations
   (`data/waymo_top_lidar_inclinations.npy`). A spoof point behind the first real return on its ray is dropped;
   a spoof point in front removes the real returns behind it (with probability 0.01 it becomes a dual return and
   the first real return survives); <= 2 returns per ray. ~4-5% of attacked frames end with 0 spoof points.
6. Intensity (spoof points only; same random stream in every mode, so geometry is identical across modes):
   adaptive = lognormal draw from the band of the spoof's mean range (`data/range_conditioned_stats.npy`, raw
   Waymo scale), clipped to [p1, p99], then **tanh**; attenuated = 0.3221 for every point (env
   `ATLAS_INTENSITY_MODE=attenuated ATLAS_INTENSITY_VALUE=0.3221`); trace = recorded values. Elongation is
   resampled only in adaptive mode.
7. Output per frame: `points`, `pose`, `frame_id`, `spoof_gt` (x,y,z,dx,dy,dz,yaw,class), `n_spoof_k`/`n_spoof_r`
   (points before/after raycasting), `attack_run_id`; per segment an annos pkl; audits in `metrics/`.

Ablation switches (global generator): `--positioning relative`, `--object_mode per_frame` (new random trace each
frame), `--random p` (flicker), `--target_class Cyclist` (cyclist traces, IoU 0.5), `--spoof_source gt_paste
--object_bank <bank>` (paste: a bank object recorded at the same range +-2 m with >= 5 points, rotated to the
bearing, ALL points kept with real intensities, no raycast filter, scene points inside the box removed).

## Point removal (generators/gen_removal_cvpr.py + removal.py)
ATLAS A-HFR removal: a target vehicle is selected (desired range 60 m, min 6 m) and tracked with a velocity model
(max step 6 m); a servo-limited sector (width W deg, elevation 16 deg, max azimuth step 15 deg/frame, centre
jitter 0.05 deg) is aimed at it and points inside are removed with probability PRR(W) = 0.97 / 1.0 / 0.99 /
0.98 / 0.90 / 0.90 for W = 10..60; a range-dependent firing gate decides whether the attack fires. All random
streams are hash(seed, segment, local frame, stream name), independent of W, so the six widths attack the same
frames and targets. `--pairing_reference_metrics` re-checks segment selection, schedule, target identity/box and
firing against the matched baseline and fails on any mismatch. Output frames carry `removal_meta`; the annos
records carry `active_this_frame`.

## Point-loss baselines (generators/generate_data_adverse_baselines.py)
Same schedule, target tracking and firing gate as ATLAS removal, but scene-wide, non-adversarial corruption:
`uniform` (each point dropped with p), `ring` (N of 64 elevation layers dropped; ring-16 is a subset of ring-32),
`matched` (exactly as many points as ATLAS would remove that frame, uniformly from the whole scene).

## Gaussian control (generators/generate_gaussian_control.sh)
On exactly the attacked frames of a paired ATLAS run, add N(0, sigma^2) to every point's xyz (no points added or
removed) and copy the spoof boxes; expected to create no phantoms.

## Banks (banks/)
- `traces_{easy,medium,hard}.pkl` (shipped originals in `data/`): first 300 Vehicle boxes, in val loader order,
  whose inside-point count is in [100,200] / [300,500] / [500,1000]; `build_trace_banks.py` rebuilds them (same
  traces and point sets; point row order can differ).
- `traces_val_cyclist_hard.pkl` (shipped): cyclist trace bank (100-300 points; 58 distinct cyclists).
- `object_bank_val.pkl` (vehicle 1.8 GB, cyclist 36 MB; build with `build_val_object_bank.sh`): every labelled
  object (every 5th frame for vehicles, every frame for cyclists) of the val segments NOT among the 50, with its
  points, box, segment, frame and range, up to 60 m, >= 5 points.

## Most efficient way to get a dataset
1. Already have the 195-segment version made with these CVPR generators and the same settings? -> `subset_50.py`.
2. Otherwise `generate.sh <name>`: ~2 h (global) / ~3.3 h (relative) on 8 cores; run many in parallel (one job
   per dataset); peak RAM ~56 GB (global) / ~132 GB (relative); ~40 GB output each.
3. Order: banks -> injection/paste/cyclist (any order) ; baselines -> removal; ATLAS run -> its Gaussian control.
4. Generation is resumable and deterministic; `verify_dataset.py` must pass before evaluating.
