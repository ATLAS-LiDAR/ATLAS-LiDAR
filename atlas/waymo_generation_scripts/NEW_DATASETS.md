# New datasets for the resubmission (generate on NCSA Delta)

This folder is a self-contained copy of the ATLAS Waymo kit (`ATLAS-LiDAR/atlas/waymo_generation_scripts`) plus
the generators for every paper dataset that did not exist on PACE as of 2026-11-05. Everything is CPU-only.
All datasets use the same 50 Waymo val segments (seed 42), the same 32-frame clean/attack schedule and the same
output layout as the existing ATLAS datasets, so the existing eval scripts run on them unchanged.

## 1. Setup (once)
1. Edit `config.sh`: `WAYMO_ROOT`, `OPENPCDET`, `OUT_ROOT`, `SLURM_ACCOUNT`, `SLURM_PARTITION`, `CONDA_ENV`.
   Paste/cyclist banks go under `$OUT_ROOT/_banks/val_object_bank` unless `BANK_DIR` is set.
2. The scene-level sets copy their target vehicles from the ATLAS removal dataset `atlas_removal_az20_seed42`.
   If `$OUT_ROOT/atlas_removal_az20_seed42` is already there (e.g. transferred from PACE), nothing else is needed;
   otherwise `submit_new.sh` generates `baseline_matched_az50_seed42` and then `atlas_removal_az20_seed42` first.
3. `bash generate.sh --list-new` lists the 38 new datasets; `DRY=1 bash submit_new.sh` previews the jobs.

## 2. Run
    bash submit_new.sh                 # everything below, with dependencies
    bash submit_new.sh <name> ...      # a subset
Each job runs `bash generate.sh <name>`: resumable per segment, ends with `verify_dataset.py`.
Typical cost per dataset (8 cores): injection ~2 h (96 GB), scene-level ~1-1.5 h (64 GB), class banks ~6-10 h.
Total ~40 jobs (38 datasets + banks + reference), ~80-100 job-hours x 8 cores (~700-800 core-hours).

## 3. The datasets
| Name | Paper row | What it is |
|---|---|---|
| `abl_cyc_atlas_{global,relative}_{easy,medium}{,_att}_seed42` (8) | class-wise table, cyclist easy/medium | ATLAS cyclist phantom (same as `abl_cyc_atlas_{global,relative}_seed42`, which is the "hard" bin), traces from `cyclist3/traces_val_cyclist_{easy,medium}.pkl`; adaptive and attenuated |
| `abl_ped_atlas_{global,relative}_{easy,medium,hard}{,_att}_seed42` (12) | class-wise table, pedestrian | ATLAS pedestrian phantom, traces from `pedestrian/traces_val_pedestrian_*.pkl`; IoU 0.5 like cyclists |
| `global_{medium,hard}_att_view_seed42` (2) | viewpoint-consistent phantom | `global_*_att_seed42` but every frame each spoof point is moved perpendicular to its ray by N(0, 0.05 m) (`ATLAS_VIEW_TANGENT`): successive sweeps hit different surface spots of the same object, as for a real object. Frame-paired with the base set (separate RNG stream) |
| `spray_l{1,2}_seed42` | road spray | point clusters trailing vehicles faster than 15 m/s; level 1 (~1 mm water film) / 2 (2-3 mm) |
| `crosstalk_{006,010}_seed42` | crosstalk | 0.6 % / 1.0 % of points displaced by N(0, 3 m) per axis (Robo3D cross-talk) |
| `fog_a{002,006}_seed42` | fog | extinction alpha 0.02 / 0.06 (visibility ~150 / 50 m) |
| `blackout_{3,10}_seed42` | sensor blackout | 3 / 10 consecutive empty frames per attack block |
| `stuck_{4,10}_seed42` | stuck frames | one frame repeated 4 / 10 times per attack block |
| `delayed_{1,3}_seed42` | delayed frames | every attack frame shows the sweep from 1 / 3 frames earlier |
| `dropped_{025,050}_seed42` | dropped frames | each attack frame dropped with p = 0.25 / 0.5, replaced by the last received sweep |
| `pose_noise_{low,high}_seed42` | pose noise | ego-pose noise rot/trans sigma 0.04 / 0.004 and 0.10 / 0.010 for history alignment (see 5) |

Trace-intensity medium sets are generated on PACE (`global_medium_trace_50_seed42`, `rel_fixed_medium_trace_50_seed42`).
Rain and snow are NOT included (no generator; they appear only in `resub_perturbations.tex`, not in the spec).
The mitigation held-out set is not included (not defined in the paper yet).

## 4. Order and dependencies (handled by submit_new.sh)
1. `banks/build_class_trace_banks.sh` -> pedestrian easy/medium/hard and cyclist easy/medium trace banks.
   Bins (points on the object, all ranges <= 60 m), shared by both classes so cyclist "hard" equals the
   existing `traces_val_cyclist_hard.pkl` band: easy 30-60, medium 60-100, hard 100-300. Check
   `bank_summary.json` (distinct objects per bin; points at 13-22 m) before relying on a bin.
2. Pedestrian / cyclist injection sets (after 1).
3. Scene-level sets (after `atlas_removal_az20_seed42` exists).
4. Viewpoint sets (no dependency).

## 5. Evaluation notes (read before running evals)
* Injection sets (pedestrian, cyclist, viewpoint): score like every injection set (EPR / ASR, IoU 0.5 for
  pedestrian and cyclist). Evaluate with the usual launchers.
* Scene-level sets: every frame carries `spoof_gt` / `removal_meta` copied from `atlas_removal_az20_seed42`, so they
  are scored with the REMOVAL scorer (conditional hiding rate of the tracked vehicle) on the same frames and targets
  as ATLAS removal and the point-loss baselines, plus collateral damage (scene mAP). `perturb_meta` per frame records
  what was done (spray cluster boxes, counts).
* Pose noise changes no points. The evals align history sweeps with the poses from the OpenPCDet infos, so the noisy
  poses are applied through `ATLAS_POSE_OVERRIDE=$OUT_ROOT/pose_noise_*_seed42/metrics/pose_noise.pkl`, read by
  `eval/eval_cp4f.sh` and `eval/eval_msf.sh` in THIS folder (patched). Single-frame CP-SF is unaffected by design.
  PTT's own proposal tracking is not patched (it consumes CP4F proposals made with the override).
* Temporal faults are stream faults: the faulty sweep is what the sensor delivers at that frame, so it is both the
  current frame and (later) history; single-frame models are affected too (as on a real car).

## 6. Modelling choices to state in the paper (and caveats)
* Spray: simplified model, not a physics simulator: Poisson count per vehicle ~ base x (speed/20)^2 x (15/range)^2
  (base 60 / 250 points for level 1 / 2), cloud of length 2 + 0.15 x speed m behind the rear bumper, widening and
  rising with distance, intensity U(0, 0.05). **Check how many vehicles exceed 15 m/s** in the 50 segments
  (`n_added` in `metrics/frames/*.csv`): Waymo is mostly urban, so few may qualify. If almost none do, set
  `SPRAY_MIN_SPEED` (env, read by `datasets.sh`) lower, regenerate, and change the paper's threshold accordingly.
* Fog: Hahner-style simplification: intensity x exp(-2 alpha R); returns below a detection floor (0.005, tanh
  scale) are lost and half of them return a soft back-scatter point at R x Beta(2, 5) (>= 1.5 m) with low intensity.
  Not the full Hahner et al. simulator; if their code is available, it can replace `scene_perturbations.fog`.
* Crosstalk: displaced points (not added), sigma 3 m per axis.
* Blackout: an empty frame is represented by one dummy ground point 70 m ahead (pcdet cannot voxelize zero points).
* Pose-noise units: rotation sigma in radians (about x, y, z), translation in metres, as in the spec table;
  **verify against the cited corruption benchmark** before the paper states them.
* All temporal faults act on the attack blocks only; the start of a blackout / stuck run is drawn per block.

## 7. Transfer back / where results go
Each dataset is `$OUT_ROOT/<name>/{dataset,annos,metrics,generation.log}` (~35-40 GB per 50-segment set; ~1.4-1.5 TB for all 38 + reference + banks). Evals
write into `<name>/evaluation/` exactly as on PACE.

## 8. Smoke test on PACE (2026-11-04/05, 1 segment per operator)
* All 8 scene operators: 95/95 attack frames changed, targets copied (89 active frames), dataset check OK.
  Clean frames contain exactly the same points as the current-pipeline removal sets (e.g. atlas_removal_az40);
  point ORDER differs between any two runs because the OpenPCDet waymo config shuffles points at test time.
  (The July atlas_removal_az20 differs from the current loader in ~0.6 % of points; only its targets are used.)
* Viewpoint phantom: spoof points of frame 41 within 1 cm of frame-40 points (ego-compensated) 89 % -> 1 %;
  spoof box identical to the base set; dataset check OK.
* Spray added 0 points on that segment (no vehicle > 15 m/s): check the counts after generation (section 6).
