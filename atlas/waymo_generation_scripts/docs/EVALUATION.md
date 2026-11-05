# Evaluation: models, checkpoints, pipelines, metrics

Everything needed to evaluate a detector on the ATLAS datasets exactly as the paper does, and to add your own
detector. The scripts in `eval/` are the exact ones used for the paper (PACE paths in their EDIT blocks).

## 1. Detectors in the paper

| Model | Repo / config | Frames | Checkpoints evaluated (GT sampling and no GT sampling) |
|---|---|---|---|
| CP-SF (CenterPoint single-frame) | OpenPCDet `custom_models/centerpoint_singleframe_waymo.yaml` | 1 | e9, 18, 27, 30 |
| CP4F (CenterPoint 4-frame) | OpenPCDet `custom_models/centerpoint_multiframe_waymo.yaml` (`SAMPLE_OFFSET [-3,0]`, timestamp channel) | 4 | e9, 18, 27, 36 |
| MSF-4 / MSF-8 | MSF repo `waymo_models/msf_{4,8}frames.yaml` (2nd stage on CP4F proposals) | 4 / 8 | e2, 4, 6 |
| PTT-32 | PTT repo `waymo_models/ptt_32frames.yaml` (MPPNet + PTTHead, 2nd stage on CP4F proposals) | 32 (proposal track) | e2, 4, 6 |

- **GT sampling** = OpenPCDet's `gt_sampling` augmentation (pastes real database objects into training frames:
  Vehicle 15, Pedestrian 10, Cyclist 10 per frame); **no-GT** = same config with `DISABLE_AUG_LIST: ['gt_sampling']`.
- MSF and PTT never use GT sampling themselves; "GT"/"no-GT" means which CP4F (GT or no-GT, epoch 36) produced
  the proposals they were trained on and are evaluated with. No-GT MSF is being trained (Nov 2026).
- Removal / point-loss baselines are evaluated at the final epoch of the GT models only (CP-SF e30, CP4F e36,
  MSF-4/8 e6, PTT e6).

## 2. Inference pipelines (all read the STORED corrupted clouds, never the clean loader)

Every dataset frame dict (`dataset/<segment>.pkl`, list per frame) holds `points` (N,6: x,y,z,intensity,elongation,
lag) already attacked, the ego `pose`, `frame_id`, and `spoof_gt` (box of the spoof / removal target, or None).

- **CP-SF** (`eval/eval_cpsf.sh <run>`): run the checkpoint on each stored frame. Predictions per segment
  (`<run>/evaluation/cp_sf_atlas/<label>/predictions/<seg>_p.pkl`, one dict per frame: `boxes_lidar` (N,7+),
  `score`, `pred_labels`), plus metrics.json (Waymo L2) and summary.pkl.
- **CP4F** (`eval/eval_cp4f.sh <run> gt|nogt|all`): for frame t rebuild the 4-sweep input [t, t-1, t-2, t-3] from
  the stored clouds (ego-pose transform into frame t + 0.1 x lag timestamp column); windows with no modified frame
  use the loader's native aggregation (identical by construction). Must use OpenPCDet's pcdet (the PTT fork's CP4F
  gives 40-90% more boxes). Raw CP4F boxes are scored (no MSF velocity rescale). The CP4F e36 predictions are the
  proposals for PTT/MSF of the same family, so run CP4F e36 first.
- **PTT** (`eval/eval_ptt.sh <run> gt|nogt`): proposals = that family's OpenPCDet CP4F e36 predictions of THIS
  dataset (frame_id-matched cache, checked against the CP4F's clean full-val output on unattacked windows; aborts
  on any miss). 32-frame proposal history with the -0.1 velocity scaling and ego-motion compensation. Current-frame
  points only (PTT's design). Labels `{gt,nogt}_epoch{E}`; cross-pairing labels swap the proposal family.
- **MSF** (`eval/eval_msf.sh <run> 4|8 <epochs>`): proposals ONLY from a seeded cache: `eval/seed_msf.sh <run>`
  (wraps `seed_msf_proposals.py`) writes `<run>/evaluation/msf_cp4f_proposals/gt_epoch36/` from the run's
  OpenPCDet CP4F GT e36 predictions (frame_id alignment + clean full-val gate: box count within 5%, median top-20
  score gap <= 0.01). MSF propagates proposals into the past 3/7 frames and samples points from the stored clouds
  of those frames. Frames with zero CP4F proposals get an empty prediction (counted as a failure). GT e6 writes the
  flat `msf{4,8}_cvpr/`, other epochs `msf{4,8}_cvpr/gt_epoch{E}/`. No-GT MSF: same, with the no-GT MSF checkpoint
  and a `nogt_epoch36` cache seeded from `cp4f_atlas/nogt_epoch36/predictions` (`--label nogt_epoch36`, no-GT
  CP4F full-val result.pkl as `--fullval`).
- `eval/run_chain.sh` runs several eval tasks in one Slurm job; `eval/launch_lib.sh` submits each job twice
  (H100 + L40S "twins"; the first to start on a working GPU cancels the other).

### Adding your own detector (PHiM, LION, DSVT, CenterFormer, ...)
- Single-frame model: copy the CP-SF pattern: load `dataset/<seg>.pkl`, feed `points[:, :5]` (or your feature
  set; intensity/elongation are tanh-scaled exactly as pcdet's loader provides) for every frame, save predictions
  in the same per-segment format. Then score with `eval/scoring/atlas_metrics.py` (injection) or
  `eval/scoring/atlas_removal_eval_only.py` (removal).
- Multi-frame / streaming model: rebuild your history from the stored corrupted frames of the same segment (as
  CP4F above), never from the clean dataset, or the attack disappears from past frames.
- Two-stage on external proposals: seed proposals from the run's own predictions of your RPN (as MSF/PTT).

## 3. Metrics

**Injection ASR** (`eval/scoring/atlas_metrics.py`): for each attacked frame (`spoof_gt` present) and cutoff
t in {0.1, 0.3, 0.5, 0.7}: keep predictions with score >= t, take the one with the highest rotated 3D IoU to the
spoof box (any class); success if it is the target class (Vehicle for car spoofs, Cyclist for cyclist spoofs) with
IoU >= 0.7 (Vehicle) / 0.5 (Ped, Cyclist). ASR = successes / attacked frames, pooled; frames whose spoof lost all
points count as failures. Also: any-class hits, per-segment ASR, phantom score distribution, best-IoU distribution.

**Removal ASR** (`eval/scoring/atlas_removal_eval_only.py`, the rebuttal scorer): target frames = frames whose
annos record (`spoof_annos[i]`) has `active_this_frame`; target box = `spoof_gt` (10 values: box, vx, vy, class
LAST). The target is detected if, among predictions **of the target class**, the best 3D IoU >= 0.7 (0.5 ped/cyc).
ASR = hidden / active frames. The scorer also reports score cutoffs 0.1-0.7, target-IoU cutoffs 0.3/0.5/0.7,
recall vs threshold, confidence drop and box drift vs the clean full-val predictions (`--clean <result.pkl>`),
and target-region false positives. Do NOT score removal with the injection rule.

**Clean accuracy**: Waymo L2 mAP/mAPH (official evaluator via OpenPCDet) over all frames of the dataset, averaged
over Vehicle/Pedestrian/Cyclist. Clean vehicle recall @ IoU 0.7 on the clean frames (for the GT-effect regression).

**Uncertainty**: paired segment bootstrap (resample the 50 segments; same frames in both conditions), 95% CI.

**GT-sampling effect** (`eval/scoring/score_extra.py`): OLS over all checkpoints, ASR(0.1) = a + b x clean recall
+ c x GT; c = ASR points at equal clean accuracy.

**Lifetime / k-history** (`eval/scoring/history_curve.py`): onset horizon h = min(frames since attack start, W)
(frames from the W-th on pool at h = W; W = 4 CP4F/MSF-4, 8 MSF-8, 32 PTT), decay h = W + d for the d-th clean
frame after the attack, scored against the last spoof box held fixed in the world.

**Clean false positives** (`eval/scoring/clean_fp.py`): Vehicle boxes with BEV IoU < 0.1 to every annotated box,
per frame, by score cutoff and range.

**Per-frame export** (`eval/scoring/export_frames.py`): one parquet row per (checkpoint, attacked frame) with range,
kept spoof points, ego speed, window position, and IoU/label/score/hit at each cutoff; reproduces ASR exactly.

## 4. Cost (50 segments, measured on L40S/H100)

| Job | GPU-h |
|---|---|
| CP-SF, 8 checkpoints | ~2.3 (two datasets per job: ~4.6 h, max 5.7) |
| CP4F e36, GT + no-GT | ~1.9 (max 2.2) |
| CP4F e9/18/27, one family | ~2.7 (max 3.0) |
| PTT, 3 epochs, one family | ~1.6 (6 epochs max 3.05) |
| MSF-4 / MSF-8, per checkpoint | ~1.3 / ~2.2 |

Full GT vs no-GT sweep of one dataset (all five models at the epochs above): ~35 GPU-h.

## 5. Output layout

`<run>/evaluation/<det>/<label>/{predictions/<seg>_p.pkl, metrics.json, summary.pkl, frame_records.pkl}` with
`<det>` in `cp_sf_atlas`, `cp4f_atlas`, `ptt32_atlas`, `msf4_cvpr`, `msf8_cvpr`; `atlas_metrics.json` per run
after scoring. `eval/launchers/` holds the PACE launchers we used (GT-vs-no-GT wave, MSF sweep, view builder) as
references for dependency ordering (CP4F e36 -> PTT / MSF seeding).
