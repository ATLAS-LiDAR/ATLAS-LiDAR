# The 63 datasets (all on the same 50 Waymo val segments, seed 42), updated 2026-11-04

All rows use the same 50 Waymo validation segments (`data/segments_50_seed42.txt`), the same 32-frame clean/attack
schedule (clean 0-31, attack 32-63, clean 64-95, ...) and car spoofs/targets unless stated. Attenuated = every spoof
point gets constant intensity 0.3221; adaptive = range-matched lognormal draw (tanh-scaled). "View" = on PACE, a
symlinked 50-segment subset of a 195-segment split (identical frames; see `subset_50.py`). Every name is a key in
`datasets.sh` (rows 1-12: the kit name drops the PACE `sub50_` prefix, e.g. `global_easy_seed42`; row 63 is built by
`generators/generate_gaussian_control.sh`, not `datasets.sh`) (`spec <name>` prints its generator arguments; `bash generate.sh <name>` builds it). Generation details:
`GENERATION.md`. Per-model eval progress: `EVAL_STATUS.md`. All 63 are generated on PACE.

Deleted / not in the paper (do not regenerate): combined-jitter sets, calibrated, saturated (max intensity), attenuated
v0, tanh-patch diagnostic, unpaired rebuttal removal_20deg/50deg, all NeurIPS-era injection datasets (intensity bug).

| # | Dataset | Description |
|---|---|---|
| **Point injection: main splits** | | |
| 1 | `sub50_global_easy_seed42` (kit: `global_easy_seed42`) | Car spoof fixed in the **world** (ego drives past it), 100-200 trace points, adaptive (range-matched) intensity |
| 2 | `sub50_global_medium_seed42` (kit: `global_medium_seed42`) | Same, 300-500 points |
| 3 | `sub50_global_hard_seed42` (kit: `global_hard_seed42`) | Same, 500-1000 points |
| 4 | `sub50_rel_fixed_easy_seed42` (kit: `rel_fixed_easy_seed42`) | Car spoof fixed **relative to the ego** (15-20 m ahead, moves with it), 100-200 points, adaptive |
| 5 | `sub50_rel_fixed_medium_seed42` (kit: `rel_fixed_medium_seed42`) | Same, 300-500 points |
| 6 | `sub50_rel_fixed_hard_seed42` (kit: `rel_fixed_hard_seed42`) | Same, 500-1000 points |
| 7 | `sub50_global_easy_att_seed42` (kit: `global_easy_att_seed42`) | Row 1 with attenuated intensity (same frames/points/positions) |
| 8 | `sub50_global_medium_att_seed42` (kit: `global_medium_att_seed42`) | Row 2, attenuated |
| 9 | `sub50_global_hard_att_seed42` (kit: `global_hard_att_seed42`) | Row 3, attenuated |
| 10 | `sub50_rel_fixed_easy_att_seed42` (kit: `rel_fixed_easy_att_seed42`) | Row 4, attenuated |
| 11 | `sub50_rel_fixed_medium_att_seed42` (kit: `rel_fixed_medium_att_seed42`) | Row 5, attenuated |
| 12 | `sub50_rel_fixed_hard_att_seed42` (kit: `rel_fixed_hard_att_seed42`) | Row 6, attenuated |
| **Point injection: post-rebuttal ablations** | | |
| 13 | `abl_car_atlas_hard_global_seed42` | Hard car trace, world-fixed, adaptive. Paired reference for rows 14-15 (same generator and draws) |
| 14 | `abl_car_atlas_hard_flicker_seed42` | Row 13, but the spoofer fires on each attack frame with p = 0.5 (imperfect spoofer; scored on fired frames) |
| 15 | `abl_car_atlas_hard_randobj_seed42` | Row 13, but a different random car trace every frame (shape not consistent over time) |
| 16 | `abl_car_atlas_hard_global_att_seed42` | Row 13, attenuated |
| 17 | `abl_car_atlas_hard_flicker_att_seed42` | Row 14, attenuated |
| 18 | `abl_car_atlas_hard_randobj_att_seed42` | Row 15, attenuated |
| 19 | `abl_cyc_atlas_global_seed42` | **Cyclist** trace spoof (58-cyclist bank), world-fixed, adaptive; target class Cyclist (IoU 0.5) |
| 20 | `abl_cyc_atlas_relative_seed42` | Cyclist, ego-fixed, adaptive |
| 21 | `abl_cyc_atlas_flicker_seed42` | Cyclist, world-fixed, fires with p = 0.5, adaptive |
| 22 | `abl_cyc_atlas_randobj_seed42` | Cyclist, new random cyclist trace every frame, adaptive |
| 23 | `abl_cyc_atlas_global_att_seed42` | Row 19, attenuated |
| 24 | `abl_cyc_atlas_relative_att_seed42` | Row 20, attenuated |
| 25 | `abl_cyc_atlas_flicker_att_seed42` | Row 21, attenuated |
| 26 | `abl_cyc_atlas_randobj_att_seed42` | Row 22, attenuated |
| 27 | `abl_car_paste_global_seed42` | **Paste** (most idealized spoof): a real car's points + real intensities copied from another val segment (recorded at the same range), all points kept, no raycasting filter, as OpenPCDet GT sampling does; world-fixed |
| 28 | `abl_car_paste_relative_seed42` | Car paste, ego-fixed |
| 29 | `abl_car_paste_flicker_seed42` | Car paste, fires with p = 0.5 |
| 30 | `abl_car_paste_randobj_seed42` | Car paste, new real car every frame |
| 31 | `abl_cyc_paste_global_seed42` | Cyclist paste, world-fixed |
| 32 | `abl_cyc_paste_relative_seed42` | Cyclist paste, ego-fixed |
| 33 | `abl_cyc_paste_flicker_seed42` | Cyclist paste, p = 0.5 |
| 34 | `abl_cyc_paste_randobj_seed42` | Cyclist paste, new real cyclist every frame |
| **Point injection: spoofer position jitter (all attenuated; Sato NDSS'24 along-ray error)** | | |
| 35 | `global_easy_att_jpt_50_seed42` | Per-point jitter only: each spoof point moved along its laser ray by N(0, 0.10 m) (object looks fuzzy); global easy |
| 36 | `global_medium_att_jpt_50_seed42` | Per-point only, global medium |
| 37 | `global_hard_att_jpt_50_seed42` | Per-point only, global hard |
| 38 | `rel_fixed_easy_att_jpt_50_seed42` | Per-point only, relative easy |
| 39 | `rel_fixed_medium_att_jpt_50_seed42` | Per-point only, relative medium |
| 40 | `rel_fixed_hard_att_jpt_50_seed42` | Per-point only, relative hard |
| 41 | `global_easy_att_jfr_50_seed42` | Per-frame jitter only: whole object shifted along the line of sight by one N(0, 0.35 m) offset per frame (shape intact, position wobbles); global easy |
| 42 | `global_medium_att_jfr_50_seed42` | Per-frame only, global medium |
| 43 | `global_hard_att_jfr_50_seed42` | Per-frame only, global hard |
| 44 | `rel_fixed_easy_att_jfr_50_seed42` | Per-frame only, relative easy |
| 45 | `rel_fixed_medium_att_jfr_50_seed42` | Per-frame only, relative medium |
| 46 | `rel_fixed_hard_att_jfr_50_seed42` | Per-frame only, relative hard |
| **Point injection: other intensity assumptions** | | |
| 47 | `sub50_global_easy_full_trace_seed42` | Spoof keeps the trace's own recorded intensities (replay attacker), global easy |
| 48 | `sub50_global_hard_full_trace_seed42` | Trace intensities, global hard |
| 49 | `sub50_rel_fixed_easy_full_trace_seed42` | Trace intensities, relative easy |
| 50 | `sub50_rel_fixed_hard_full_trace_seed42` | Trace intensities, relative hard |
| **Point removal (ATLAS A-HFR, frame-paired across widths and with the baselines)** | | |
| 51 | `atlas_removal_az10_seed42` | Attacker deletes points in a 10° azimuth × 16° elevation sector tracking one target vehicle (removal prob. 0.97 inside the sector, servo-limited aim) |
| 52 | `atlas_removal_az20_seed42` | 20° sector, removal prob. 1.0 |
| 53 | `atlas_removal_az30_seed42` | 30° sector, 0.99 |
| 54 | `atlas_removal_az40_seed42` | 40° sector, 0.98 |
| 55 | `atlas_removal_az50_seed42` | 50° sector, 0.90 |
| 56 | `atlas_removal_az60_seed42` | 60° sector, 0.90 |
| **Removal baselines (non-adversarial point loss, same frames and targets as ATLAS removal)** | | |
| 57 | `baseline_matched_az20_seed42` | Deletes exactly as many points as ATLAS az20 would that frame, but uniformly from the whole scene (isolates spatial targeting) |
| 58 | `baseline_matched_az50_seed42` | Same, matched to az50 |
| 59 | `baseline_uniform_50_seed42` | Scene-wide random dropout: each point deleted with p = 0.5 on attack frames |
| 60 | `baseline_uniform_90_seed42` | Scene-wide random dropout, p = 0.9 |
| 61 | `baseline_ring_16_seed42` | Beam/ring dropout: 16 of the 64 elevation layers deleted per attack block |
| 62 | `baseline_ring_32_seed42` | Ring dropout, 32 layers (superset of ring-16) |
| **Control** | | |
| 63 | `gaussian_sigma_008_global_easy_seed42` | Injection control from the rebuttal: Gaussian position noise σ = 0.08 instead of a spoof; expected to create no phantoms |
