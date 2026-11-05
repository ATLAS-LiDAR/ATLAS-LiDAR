# Open questions for the authors (Waymo side, 2026-11-05)

Each question blocks or changes work before the 11/14 results freeze. Answer inline (replace `ANSWER:` with the decision)
and push; the PACE side reads this file. **Default** = what happens if nobody answers by the date given.
Context: `PROGRESS_ESTIMATE.md` (schedule, resources, training estimates), `WORK_PLAN.md` (table → datasets),
paper source = `ATLAS-LiDAR/overleaf`.

## A. Training (main paper; decide by 11/6, because GPU queues are long)

**A1. Tier C fine-tune recipe** (fade figure, mitigation table; the paper has a TODO here).
- How many epochs, and what LR / schedule? (Suggestion: the last 1/6 of the original schedule, constant LR = final LR of the
  original one-cycle schedule.)
- Which detectors: CP-SF and CP4F only, or also MSF-4/8 and PTT?
- Do MSF and PTT second stages use GT sampling in their own training at all? If not, "fine-tune without GT sampling"
  means only the proposal network (CP4F) changes for them.
- ANSWER:
- Default: CP-SF and CP4F only, suggested recipe; MSF/PTT listed as future work.

**A2. MSF-8 no-GT** (GT-sampling table, MSF-8 row). Delta's job has been stuck for days; PACE L40S estimates a start ~11/10,
giving e2 (maybe e4) by 11/13.
- Submit a second copy on PACE (keep Delta queued, cancel whichever starts second)?
- If only e2/e4 exist by the freeze: report them with a note, or drop the MSF-8 no-GT column?
- ANSWER:
- Default: submit on PACE; report whatever checkpoints exist, with a note.

**A3. MSF-4 no-GT on Delta**: ~~is training finished?~~ RESOLVED 11/5: finished (6 epochs), on PACE, evals submitted
(`PROGRESS_ESTIMATE.md` section 10). Open: Delta produces clean val `result.pkl` for e2/e4 and transfers them to
`transfers/msf4_nogt_delta/eval_epoch{N}_val/`.
- Should MSF-4 (GT + no-GT) be added to the class dose-response sets (about 20 evals)?
- ANSWER:

**A4. CP-SF paste-count 5 / 30** (dose-response, paste-count part): still wanted given the queue? 0 and 15 exist.
- ANSWER:
- Default: submit after MSF-8; drop the 5/30 points if not done by 11/12.

**A5. Mitigations needing new code** (raycast-consistent GT sampling; adversarial training with phantoms as negatives).
Is anyone writing them this week? What is the "held-out setting never used for training" in `tabs/resub_mitigation.tex`?
- ANSWER:
- Default: both moved to future work; mitigation table = score filter (≥ 0.7) + Tier C rows.

## B. Metric and table definitions

**B1. GT effect.** The paper defines Δ "at matched accuracy" as the GT coefficient of a regression over all checkpoints
(`tabs/resub_gt_sampling.tex`, `tabs/resub_fidelity_rank.tex`). Every-checkpoint evals exist for CP-SF / CP4F / PTT on the
injection sets, but the new scene sets (spray, crosstalk, fog) and removal no-GT are evaluated at the final epoch only.
- Keep the regression for the GT-sampling table and use the final-epoch GT − no-GT difference for the fidelity ladder?
- ANSWER:
- Default: yes (regression where all checkpoints exist; final-epoch difference elsewhere, stated in the caption).

**B2. Checkpoints for "Tier B".** PTT runs at e2/4/6 (not e1–6), MSF at e2/4/6, CP-SF e9/18/27/30, CP4F e9/18/27/36. OK?
- ANSWER:

**B3. Dose-response figure, class part.** Which pedestrian / cyclist sets go in the main figure? Running now: the
**attenuated** sets (pedestrian global/relative easy/medium/hard; cyclist easy/medium + existing hard). Adaptive
(non-attenuated) variants go to the supplement class-wise table after submission.
- ANSWER:

**B4. Main-table intensity**: attenuated (the paper's reference) as the headline, adaptive reported as well?
- ANSWER:

**B5. Scoring design choices** (`eval/scoring/paper_metrics.py` computes all of them; pick one per table):
fired vs all-valid attack frames; argmax vs any-match; τ_f (precision 0.9 on clean frames) vs fixed 0.1/0.3/0.5/0.7; track
variant (`block_filled|onset_from_block` etc.).
- ANSWER:
- Default: fired, argmax, τ_f; track variant `block_filled|onset_from_block`.

## C. Spray, weather, and table-less datasets

**C1. Spray threshold.** `tabs/resub_perturbation_spec.tex` says "> 15 m/s" with no citation. On the 50 segments only 15
segments and 2% of attack-frame vehicle boxes qualify. Generated: spray_l1/l2 at 15 m/s (as in the paper) and
spray_l1_v8 (8 m/s, ~6% of boxes) as an ablation.
- Cite a source for 15 m/s, or call it our choice? Report spray on all segments, the affected subset, or both?
- The spec says water film "~1 mm and 2–3 mm"; the generator implements two density levels (base 60 / 250 points,
  ∝ speed², ∝ 1/range²). How should the text describe that mapping?
- ANSWER:

**C2. Results tables for datasets that have none.** Generated, but no results table exists in the main paper or the
supplement: temporal faults (blackout, stuck, delayed, dropped: 8 sets), pose noise (2), viewpoint-consistent phantom (2).
They are described in `sec/5_evaluations.tex:37` and the spec table.
- Add a supplement table (evaluated 11/14–11/23), or cut them from the text?
- ANSWER:
- Default: supplement table, evaluated after submission.

**C3. Rain and snow** appear in the paper text but there is no generator. Cut?
- ANSWER:

## D. Text that does not match what was run (main paper)

**D1.** CP-SF is trained for 30 epochs on every 5th frame; the paper says 36 epochs (evaluated at 9/18/27/36).
**D2.** RQ4 says CP-SF vs CP4F have an "identical recipe apart from the number of input sweeps"; they differ in training
frames (1/5 vs all), epochs (30 vs 36) and CP4F batch (GT 8 vs no-GT 16; a batch-8 no-GT control is training).
**D3.** Waymo trace / cyclist banks come from val segments excluding the 50 evaluation segments, not from the training split.
- Who updates the text, and how (soften vs re-run)? A batch-8 control result is expected ~11/9.
- ANSWER:

## E. Detectors

**E1. PHiM**: is it in the main table? Who runs its inference (PACE or Delta), and on which sets? `WORK_PLAN.md` section 3
explains how to score it.
- ANSWER:
