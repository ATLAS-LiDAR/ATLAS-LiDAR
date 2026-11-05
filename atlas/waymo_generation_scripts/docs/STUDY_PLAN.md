# CVPR'27 study plan and status (updated 2026-11-04)

## Protocol decisions
- **One segment set everywhere**: the same 50 Waymo val segments (seed 42) for every injection, removal and
  baseline study. No 195-segment headline table any more; the 6 x 195 splits are only a source for 50-segment
  subsets.
- **Every point-injection study is a GT-sampling vs no-GT-sampling study**: CP-SF e9/18/27/30, CP4F e9/18/27/36,
  PTT e2/4/6 and MSF-4/8 e2/4/6, each for the GT and the no-GT family (no-GT MSF trained on NCSA Delta).
- **Removal and point-loss baselines**: final epoch, GT models only (CP-SF e30, CP4F e36, MSF-4/8 e6, PTT e6).
- One jitter component per dataset (per-point OR per-frame), never both (combined sets were confounded and deleted).
- Each study differs from the reference only in the factor it studies: intensity mode, jitter component, point
  budget, object class, spoof source, schedule, removal width, or baseline type.

## Dataset inventory (63 on the 50 segments)
| Group | Count | Status (PACE) |
|---|---|---|
| Main injection, adaptive + attenuated, 6 settings each | 12 | done (subsets of the 195-segment splits) |
| Trace intensity, 4 settings | 4 | done |
| Attenuated + per-point jitter, 6 settings | 6 | generated; evals queued |
| Attenuated + per-frame jitter, 6 settings | 6 | generated; evals queued |
| Car ATLAS ablations (global, flicker, random object) x {adaptive, attenuated} | 6 | done |
| Cyclist ATLAS ablations (global, relative, flicker, random object) x {adaptive, attenuated} | 8 | done |
| Paste (car / cyclist x global, relative, flicker, random object) | 8 | done |
| ATLAS removal az10-60 | 6 | all 6 generated (paired); final-epoch GT evals queued |
| Point-loss baselines (uniform 50/90, ring 16/32, matched az20/50) | 6 | done |
| Gaussian-noise control | 1 | done |

Deleted / superseded (do not regenerate): calibrated intensity, saturated intensity (1.0), attenuated v0
(per-library constants), combined jitter J1/J2/J3, the tanh-patch diagnostic, unpaired removal_20deg/50deg, every
NeurIPS-era injection dataset (intensity bug, below).

## Known history and caveats (so you do not rediscover them)
- **NeurIPS intensity bug**: the old generators drew spoof intensities from fits on the RAW Waymo scale while scene
  points are tanh-scaled; spoofs closer than 24 m got out-of-range intensities. CP-SF rejected those spoofs, CP4F
  less so, which produced the NeurIPS "CP4F more vulnerable" result. Fixed in the CVPR generators (tanh).
- CP4F must be run with OpenPCDet's pcdet; the PTT repo's pcdet fork gives 40-90% more CP4F boxes. July rebuttal
  removal/baseline PTT and MSF evals computed CP4F inside the forks and are superseded.
- The GT-PTT epochs come from two runs (e1-2 from a 6-epoch GT run, e3-6 from the February run); judged not to
  change results materially.
- GT and no-GT training recipes differ in batch size (e.g. CP4F 8 vs 16); single training run per family.
- Removal datasets store a 10-value `spoof_gt` (class last); injection stores 8 values (class at index 7).
- Rebuilt trace libraries match the originals as point sets but not in row order; use the shipped originals for
  byte-exact data.

## Planned additional ablations (to generate/evaluate on NCSA Delta, clean scenes, 50 segments)
| Ablation | Levels | Build | Metric |
|---|---|---|---|
| Road spray (clusters behind moving vehicles) | water film ~1 mm and 2-3 mm, vehicles > 15 m/s | new generator | phantom rate in the spray region + L2 |
| LiDAR crosstalk | 0.6% and 1.0% of points displaced (Robo3D) | new generator | L2 drop, phantom rate |
| Frustum / context-aware placement (spoof in line with a real object) | 1 | injection generator variant | injection ASR + target hidden/shifted |
| Fog (Hahner simulator) | alpha 0.02 (~150 m) and 0.06 (~50 m) | port of released simulator | L2 drop, phantom rate |
| Sensor blackout | 3 and 10 empty frames | copied dataset | real-vehicle recall over time, recovery frames |
| Stuck frames | 4 and 10 repeats | copied dataset | same |
| Delayed frames | 1 and 3 frames | copied dataset | same |
| Pose noise (stacking) | Dong'23 rotation/translation 0.04/0.004 and 0.10/0.010 | edited poses | same |
| Dropped history frames | p = 0.25 and 0.5 | eval-time hook (multi-frame models only) | same |
Already covered by existing data: beam loss (ring baselines), density loss (uniform dropout).
Sources and rationale: Dong et al. CVPR'23, Robo3D ICCV'23, Hahner ICCV'21, Sato NDSS'24, Hallyburton USENIX'22,
DejaVu, PhantomLiDAR NDSS'25, MultiCorrupt IV'24.
