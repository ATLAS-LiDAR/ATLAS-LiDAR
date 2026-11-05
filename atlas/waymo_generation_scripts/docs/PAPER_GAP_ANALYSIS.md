# Paper experiment list vs. what exists on PACE (Waymo side), 2026-11-05

Snapshot of 11/5 morning. For the schedule, available resources on PACE and Delta, and what is feasible by the
deadline, see `PROGRESS_ESTIMATE.md` (it supersedes the "where" column here).

Source: `adversarial_lidar/paper` (ATLAS-LiDAR/overleaf @ 6a0d83c): `sec/5_evaluations.tex`, `tabs/resub_*.tex`.
Status key: DONE = data + evals exist; PARTIAL; TODO-CPU (no GPU/training needed); TODO-GPU (inference only);
TODO-TRAIN (needs training); NEW-CODE (generator/code doesn't exist yet). nuScenes rows are out of PACE scope.

## A. Text vs. reality (fix the text or the experiment)
1. **CP-SF epochs.** Paper: "CP-SF and CP4F are trained for 36 epochs ... evaluated at 9/18/27/36". Reality: CP-SF is 30 epochs (e9/18/27/30), every 5th frame (SAMPLED_INTERVAL 5); CP4F every frame, 36 epochs.
2. **RQ4 factorial claim** "CP-SF vs CP4F, identical recipe apart from the number of input sweeps" is false: training frames (1/5 vs all), epochs (30 vs 36), batch (16/16 vs 8/16). Options: soften the text, or train CP-SF on all frames for 36 epochs, GT and no-GT (the stopped `gt_sampling_alldata` run reached e4 of 30; no no-GT counterpart exists). ~35-40 h per run on 2 H100.
3. **Batch confound** is already mentioned in a TODO; the batch-8 no-GT CP4F rerun is queued (job 15121643).
4. **Rain / Snow** rows in `resub_perturbations.tex` are not in the perturbation spec (and snow was dropped). Remove or add to the spec.
5. **Cyclist library** says "58 cyclists on W" and the class-wise table wants easy/medium/hard; the existing Waymo cyclist sets use one (hard, 100-300 pt) bin only.
6. "Trace libraries are built per class from the training split" is stated for nuScenes; Waymo car traces / cyclist bank come from val (excluding the 50 segments). Make sure the Waymo text says what was done.

## B. Datasets (perturbation spec table)
| Perturbation | Status | Notes |
|---|---|---|
| Car injection, 6 settings, adaptive | DONE | sub50 views |
| Car injection, 6 settings, attenuated (reference) | DONE | sub50 views |
| Car injection, trace intensity | PARTIAL | 4 of 6: **global_medium / rel_fixed_medium trace missing** (CPU gen, ~2-3 h each) |
| Per-point / per-frame jitter (attenuated) | DONE (generated) | evals queued |
| Flicker, per-frame trace (car hard, cyclist) | DONE | adaptive + attenuated |
| Naive paste (car, cyclist) | DONE | |
| Gaussian control | DONE | |
| Cyclist injection, rel/global x easy/med/hard | PARTIAL | have hard-bin global/relative only (+flicker, randobj). Need 4 more settings (easy, medium x 2), from a cyclist bank with 3 bins (bank builder supports bands) |
| **Pedestrian injection**, rel/global x easy/med/hard | NEW | no pedestrian trace bank yet (`build_val_object_bank` with Pedestrian bands), 6 datasets |
| **Viewpoint-consistent phantom** | NEW-CODE | re-render the trace per sweep so points don't repeat across frames (control for the frozen-trace cue) |
| Removal az10-60 | DONE | final-epoch evals queued |
| Budget-matched 20/50, uniform 50/90, beam 16/32 | DONE | |
| Road spray (2), crosstalk (2), fog (2) | code done; generating on PACE (11/5) |
| Temporal faults: dropped, stuck, delayed, blackout, pose noise (x2 each) | code done; generating on PACE (11/5) |

## C. Detectors / training (models table)
| Item | Status |
|---|---|
| Tier A+B CP-SF, CP4F, PTT | DONE (CP4F batch confound, see A3) |
| MSF-4f / MSF-8f GT | DONE; no-GT: MSF-4 reportedly trained on Delta, MSF-8 stuck in Delta queue |
| **PHiM (Waymo)** | nothing on PACE: recipe, checkpoints, eval pipeline all missing. Owner? |
| **Tier C** fine-tune w/o GT sampling, every Waymo detector | TODO-TRAIN. Open: epochs/LR (paper TODO); does "no GT sampling" even apply to MSF/PTT second stages (check whether their own training uses gt_sampling) |
| **Paste-count dose-response**: CP-SF with 0/5/15/30 vehicles | 0 and 15 exist; **5 and 30 need 2 CP-SF retrains** (same recipe as CP-SF GT; ~1 day each) |
| **Mitigations on CP-SF + CP4F** | fade = Tier C; **raycast-consistent GT sampling** (NEW-CODE aug + 2 retrains); **adversarial training w/ phantoms as negatives** (NEW-CODE + 2 retrains); held-out setting undefined (paper TODO) |
| Test-time mitigations: score >= 0.7 | TODO-CPU (free from stored predictions) |
| Intensity-consistency filter, sector point-drop detection | NEW-CODE, CPU on stored data/predictions |

## D. Metrics (sec. Metrics). We currently report raw ASR; the paper uses paired metrics.
| Metric | Needs | Status |
|---|---|---|
| EPR (phantom matched in attacked frame but not in the same clean frame) | clean-frame predictions per checkpoint | clean preds exist for CP-SF, CP4F, PTT (all epochs, `clean_recall/`); **missing for MSF-4/8** (and later no-GT MSF, bs8 CP4F, Tier C): TODO-GPU |
| Calibrated operating point tau_f (precision 0.9 on clean frames) | clean preds + GT | TODO-CPU |
| Conditional HR (removal) | clean-frame detection of the target | TODO-CPU once clean preds exist |
| Collateral damage C (mAP with target excluded), budget B | perturbed preds + custom evaluator | TODO-CPU |
| Effective robustness, matched-accuracy GT effect | all checkpoints | TODO-CPU |
| Track-level success (k=3 of n=5), onset latency, recovery | per-frame records | TODO-CPU |
| Dominance D | current-frame-only attack (history intervention) | needs E |
| Paired bootstrap 1000 over segments | per-segment counts | have the machinery |

## E. Inference-only experiments (TODO-GPU)
- **History interventions** (CP4F, MSF-4f, MSF-8f, PTT; injection global-medium, removal 40 deg): clean history / attacked history / copied current / clean current. CP4F variants exist in `diag_input_intervention`; MSF and PTT need the history buffer swapped in their loaders (NEW-CODE).
- Clean-frame evals for MSF-4/8 (for EPR/HR/tau_f).
- Attack lifecycle figure: CPU from per-frame records (global-medium, 40 deg); needs MSF/PTT medium evals (queued).

## F. Already running / queued (no action)
Injection GT/no-GT evals (all sets, all epochs) except MSF no-GT; removal final-epoch evals; batch-8 no-GT CP4F training.

## Suggested order with ~12 days left
1. CPU now: trace-medium generation (2 sets); test-time score threshold; tau_f / EPR / HR / track metrics on what exists; fix text items A1, A2 (soften), A4, A6.
2. GPU, cheap: MSF clean-frame evals; history interventions for CP4F (code exists).
3. Generation: pedestrian bank + 6 sets, cyclist easy/medium (4 sets); spray/crosstalk/fog/temporal faults: all now generating on PACE (11/5).
4. Training (decide what to cut): CP-SF paste-count 5/30 (2 runs), Tier C fine-tunes (5), raycast-consistent and adversarial mitigations (4 runs). That is ~11 training runs; realistically only some fit. PHiM needs an owner.

## G. Coverage count, 5 Waymo models (CP-SF, CP4F, MSF-4f, MSF-8f, PTT), 2026-11-05
Unit = one result cell per model the paper needs (GT and no-GT counted separately). "Covered" = data + evals
done or queued (incl. held jobs). Overall 383 / 733 = 52%.
| Table / figure | Covered | Main blockers |
|---|---|---|
| Main results (clean + 6 inj + 6 removal) | 65/65 | MSF clean-frame preds for EPR/HR |
| Class-wise ped + cyclist | 10/60 | pedestrian sets; cyclist easy/medium |
| Perturbations (effect + collateral, 14 rows) | 110/140 | fog, rain, snow |
| Fidelity ladder (8 rungs) + corruption ranks | 58/95 | spray, crosstalk, fog, MSF no-GT, trace medium |
| GT-sampling pairs | 24/40 | removal no-GT (+ all ckpts), MSF no-GT |
| Dose-response (class, paste count) | 8/19 | pedestrian, MSF no-GT, CP-SF 5/30 retrains |
| History x GT factorial | 14/16 | removal no-GT |
| History length + frame sweep | 17/28 | MSF no-GT, removal no-GT |
| Attack lifecycle | 10/16 | removal no-GT, MSF no-GT |
| History interventions | 8/40 | intervention code for MSF/PTT, runs |
| Temporal faults | 0/40 | generators |
| Fade (Tier C) | 0/10 | 5 fine-tunes |
| Mitigation (CP-SF, CP4F) | 26/96 | fade, raycast GT, adversarial training, filters, held-out set |
| Accuracy vs robustness (all ckpts) | 33/68 | removal at all ckpts GT + no-GT, MSF no-GT |

## Update 2026-11-05
- Plan change: every remaining dataset is generated on PACE (not Delta).
- Rain and snow: not generated, no generator (rows exist in resub_perturbations.tex only).
- Submitted: global_medium_trace_50_seed42 / rel_fixed_medium_trace_50_seed42 (CPU 15123500 / 15123510) and their
  full evals H100-only (H100_ONLY=1 in evals/launch_lib.sh; keeps L40S free for the bs8 CP4F training).
