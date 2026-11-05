# Waymo work plan for the resubmission: who generates / evaluates what (2026-11-05)

Paper source of truth: the Overleaf repo (`ATLAS-LiDAR/overleaf`): `sec/5_evaluations.tex`, `tabs/resub_*.tex`.
This file maps every Waymo table to datasets, so a new detector (PHiM) can be evaluated on exactly the right sets.

## 1. Where the datasets come from
| Group | Count | Where generated | Status |
|---|---|---|---|
| The 63 datasets of `DATASETS.md` (main injection, trace, jitter, car/cyclist ablations, paste, removal, baselines, Gaussian) | 63 | PACE | done |
| Trace intensity, medium: `global_medium_trace_seed42`, `rel_fixed_medium_trace_seed42` | 2 | PACE (named `*_medium_trace_50_seed42` there) | generating |
| Everything in `NEW_DATASETS.md` (pedestrian, cyclist easy/medium, viewpoint phantom, spray, crosstalk, fog, temporal faults, pose noise) | 38 | NCSA Delta (`submit_new.sh`) | generating |
| Rain, snow, mitigation held-out set | - | - | not planned (no generator / not defined) |

Every dataset: same 50 val segments (`data/segments_50_seed42.txt`), same 32-frame clean/attack schedule, same
layout `<name>/{dataset,annos,metrics}`. Names below are the kit names (`bash generate.sh --list`, `--list-new`).

## 2. Paper table -> datasets (for every Waymo detector, incl. PHiM)
Tier A = the detector's standard (GT-sampling) checkpoint, final epoch. Tier B = GT and no-GT pair, every saved
checkpoint. "6 settings" = `{global,rel_fixed}_{easy,medium,hard}`.
| Table / figure | Datasets | Checkpoints |
|---|---|---|
| Main results `resub_waymo_main` | 6 settings attenuated (`*_att_seed42`; reference) and adaptive (`*_seed42`); `atlas_removal_az{10..60}` | Tier A final |
| Class-wise `resub_waymo_classwise` | pedestrian: `abl_ped_atlas_{global,relative}_{easy,medium,hard}[_att]`; cyclist: `abl_cyc_atlas_{global,relative}_{easy,medium}[_att]` + hard = `abl_cyc_atlas_{global,relative}[_att]` | Tier A final |
| Perturbations `resub_perturbations` | ATLAS injection (6 settings), `abl_car_paste_*`, `gaussian_sigma_008_global_easy`, removal az20/az50, `baseline_{matched_az20,matched_az50,ring_16,ring_32,uniform_50,uniform_90}`, `fog_a{002,006}` | Tier A final |
| Fidelity ladder `resub_fidelity_rank` | rungs: paste (`abl_car_paste_*`), trace (`*_trace` 6 settings), adaptive, attenuated, `*_att_jpt_50`, `*_att_jfr_50`, `spray_l{1,2}`, `crosstalk_{006,010}`; corruption rows: fog, uniform 90, ring 32 | Tier A final for rho; GT effect needs Tier B (see 4) |
| GT sampling `resub_gt_sampling` | 6 settings (both intensities) + removal az10-60 (control) | Tier B, all checkpoints |
| Dose-response (class) | car / pedestrian / cyclist sets above | Tier B |
| Factorial, history length, frame sweep, lifecycle, history interventions, temporal faults | CP-SF / CP4F / MSF / PTT only (the paper's lists do not include PHiM); temporal faults: `blackout_*`, `stuck_*`, `delayed_*`, `dropped_*`, `pose_noise_*` | final, GT + no-GT |
| Fade (Tier C), mitigation | not dataset-specific (6 settings + removal 20/50) | fine-tuned models |

## 3. How to score a new detector
1. Run inference on every frame of `<name>/dataset/<segment>.pkl` (frames carry `points` (N,6: x,y,z,
   intensity, elongation, time), `pose`, `frame_id`). Write one file per segment:
   `<name>/evaluation/<det>/<label>/predictions/<segment>_p.pkl` = list over frames of OpenPCDet-style dicts
   `{name, score, boxes_lidar (N,7 or 9), pred_labels, frame_id}`.
2. Injection sets: `eval/scoring/atlas_metrics.py` (target-class hit, IoU 0.7 vehicle / 0.5 ped & cyclist, ASR at
   score 0.1/0.3/0.5/0.7, per-segment counts for bootstrap). Removal, baselines and ALL scene-level sets
   (fog, spray, crosstalk, temporal faults, pose noise): `eval/scoring/atlas_removal_eval_only.py` with the
   detector's clean val predictions as reference (targets = `spoof_gt` / `removal_meta`).
3. Paper metrics: `eval/scoring/paper_metrics.py --dataset <name dir> --pred <.../predictions> --clean <clean
   predictions of the SAME checkpoint>` writes `paper_metrics.json` next to the predictions: calibrated threshold
   tau_f (precision 0.9 on the clean frames), EPR (injection) or conditional HR (removal-type), raw ASR, track-level
   rate (3 of 5) and onset latency, collateral damage C (own AP3D, target excluded), point budget, and 95% segment
   bootstrap CIs, at tau_f and at 0.1/0.3/0.5/0.7. `--clean` = the checkpoint's predictions on unperturbed frames
   (OpenPCDet val `result.pkl`, matched by frame_id). Effective robustness is a fit across all detectors' jsons.
   Every scoring design choice (fired vs all-valid frames, argmax vs any-match, tau_f vs fixed thresholds, track
   variants) is computed and stored as a separate key; pick the paper's key when building tables.
   Validation (2026-11-05): injection path reproduces `atlas_metrics` raw ASR (CP4F GT e36, global_hard: 0.4768 vs
   0.4772). Removal path and `paired_diff.py` (joint segment bootstrap of A - B) are still being validated.
4. Multi-frame detectors: build history from the stored frames of the same dataset (not from raw Waymo), as
   `eval/eval_cp4f.sh` does; for `pose_noise_*` set `ATLAS_POSE_OVERRIDE=<dataset>/metrics/pose_noise.pkl`
   (applied to history alignment only).

## 3b. Getting the PACE datasets onto Delta (two routes)
* **Transfer** (Globus) from PACE `/storage/scratch1/9/spanse30/cvpr_datasets/<name>` (~39 GB each).
  The PACE `sub50_*` sets (main settings, trace easy/hard) are per-file SYMLINKS into 195-segment splits and Globus
  does not follow symlinks: materialise them first (`cp -rL`) or use route 2. Existing PACE evals live in
  `<name>/evaluation/` (transfer that subfolder only if Delta needs the PACE detectors' predictions).
* **Regenerate** with `bash generate.sh <kit name>` (CPU; ~2 h global, ~3.3 h relative, ~1.3 h removal/baseline;
  identical frames: every draw is keyed by seed and original frame index). Name mapping PACE -> kit:
  `sub50_<setting>_seed42` -> `<setting>_seed42`; `sub50_<setting>_att_seed42` -> `<setting>_att_seed42`;
  `sub50_<setting>_full_trace_seed42` -> `<setting>_trace_seed42`; `*_medium_trace_50_seed42` -> `*_medium_trace_seed42`;
  every other name is identical. Removal widths need `baseline_matched_az50_seed42` first (pairing check).
* Either way, a detector that only runs on Delta needs only the datasets of its tables in section 2.

## 4. Open decisions (affect how much has to be evaluated)
* Fidelity ladder "GT effect": the paper defines the GT effect as a regression over ALL checkpoints; if kept,
  every rung needs Tier B at every checkpoint; if changed to a final-epoch GT - no-GT difference, final only.
* Jitter rungs: all 6 settings, or one reference setting per jitter type.
* Main table intensity: attenuated is the reference rung; adaptive is evaluated too.
* Spray threshold (15 m/s) may leave almost no spray on urban Waymo segments (check counts, NEW_DATASETS.md 6).
* CP-SF is trained 30 epochs on every 5th frame (paper text says 36); CP4F GT/no-GT batch 8 vs 16 (a batch-8
  no-GT CP4F retrain is running on PACE as a control).

## 5. Status of the existing detectors (PACE)
See `EVAL_STATUS.md` (snapshot). In short: CP-SF / CP4F / PTT GT + no-GT and MSF-4/8 GT on all injection sets at
every checkpoint, queued or done; removal / baselines / Gaussian final-epoch GT queued; MSF no-GT waits for Delta
training; the 38 new sets are evaluated after Delta generation.
