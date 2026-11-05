# Progress estimate: what can feasibly be evaluated by 11/14 (updated 2026-11-05, evening: PACE GPU queue re-checked)

Submission 11/16; results frozen about **11/14**, so there are **9 days of compute** left. Paper coverage today: 383 of 733
required cells (52%, see `PAPER_GAP_ANALYSIS.md`).

**Priority rule: the temporal detectors (MSF-4, MSF-8, PTT) come first everywhere.** They are the novelty of the paper.
CP-SF / CP4F are the controls and fill in behind them.

## 1. What changed since this morning

- **Delta is written off for planning.** The rolling reboot cut capacity; our fair-share rank sits mid-queue; MSF-8 no-GT has
  been stuck for days; CPU generation barely moves. Anything that lands from Delta is a bonus.
- **All 39 new datasets are now generating on PACE CPU** (submitted 11/5, 13 running immediately). Delta's fixes are applied
  (bank-script path bug, restart-safe scene generator) plus the new `spray_l1_v8_seed42` (8 m/s ablation). Expected done:
  scene sets this afternoon or evening; pedestrian/cyclist sets about 3–4 h after the bank job, so by about 11/6 morning.
- **PACE GPU is congested too** (section 2b). New multi-GPU training is not reliable before 11/13 on either cluster.
  MSF-8 no-GT is the critical path for the paper's novelty and is now a race (section 3).

## 2. Compute we can count on

| Resource | State | Capacity |
|---|---|---|
| PACE GPU evals (H100 + L40S twins) | 121 evals finished in the last 24 h, about 3.6 h each | **about 100–120 evals per day** |
| PACE 4-GPU job (MSF training) | scheduler estimate: **H100 Nov 20** (after deadline), **L40S Nov 10** | L40S only; see 2b |
| PACE 8-GPU full node (CP4F training) | batch-8 retrain still pending; estimate slipped 02:20 → 12:40 → 17:12 today | at most 1 at a time; waits keep slipping |
| PACE CPU | jobs start within minutes | not a constraint |
| Storage | scratch about 3 TB free (new sets use about 1.6 TB); **project 99% full (18 GB)** | checkpoints go to cedar (3.2 TB free) |
| Delta | GPU and CPU stuck | **plan for zero** |

## 2b. PACE GPU queue (checked 11/5 evening)

| Partition | Nodes usable | Jobs pending | Our pending | Next-largest user |
|---|---|---|---|---|
| gpu-h100 | 2 of 4 (1 drained, 1 invalid) = 16 GPUs | 475 | 81 | about 41 |
| gpu-l40s | 7 of 10 (3 drained) | — | 67 | about 34 |

- Our single-GPU evals waited a **median of about 26 h** to start over the last 3 days (p90 about 37 h), on both GPU
  types. Part of that is queueing behind our own jobs.
- `sbatch --test-only` estimates assume every running job uses its full time limit, so they are pessimistic, but
  they agree with the waits we see.
- Our own evals are the largest block in both queues and use up our fair-share; a training job submitted now
  competes with them.
- The eval throughput (about 100–120 per day) has held so far, but it depends on the drained H100 nodes; if they stay
  down, plan for less.

## 3. Temporal detectors: status and plan (top priority)

| Detector | GT model | no-GT model | Evals on the existing 63 sets | Plan |
|---|---|---|---|---|
| **PTT** | done (e1–e6) | done (e1–e6) | injection GT + no-GT every checkpoint, plus proposal swap: **done**; 27 ablation/removal evals queued | evaluate the 39 new sets first (final GT; temporal-fault/pose sets also no-GT) |
| **MSF-4** | done (e2/e4/e6) | trained on Delta (per Delta's note, 28 h on 4 A100) — **confirm and copy checkpoints to PACE** | GT queued (about 9 left); no-GT not started | no-GT evals on PACE as soon as checkpoints arrive |
| **MSF-8** | done (e2/e4/e6) | **stuck on Delta** | GT queued (about 9 left); no-GT not started | **retrain no-GT on PACE now** |

**MSF-8 no-GT: a race, not a plan.** The GT run (March, PACE) took about 21 h per epoch on 2 GPUs at total batch 4,
so about 5 days for 6 epochs on 2 GPUs, or about 2.5–3 days on 4 GPUs (assumed near-linear; unmeasured).
- Delta: queued, stuck for days. PACE H100: estimated start Nov 20 (too late). PACE L40S: estimated start Nov 10.
- **Realistic outcome:** if it starts about 11/10, it has about 3 days of training before the freeze, which covers
  2–3 epochs. MSF-8 no-GT could then be reported at e2 (maybe e4) only, not e6. Not yet checked: whether MSF-8 fits in
  48 GB on L40S at 1 per GPU.
- **Option (not submitted, pending decision):** keep the Delta job queued and also submit on PACE L40S; cancel whichever starts
  second. Holding the CP-SF/CP4F evals would free our priority for it and for the MSF/PTT evals.
- **Fallback:** report the MSF GT/no-GT pair for MSF-4 only, and MSF-8 GT only; state this in the paper.

**MSF-4/8 clean predictions at e2/e4** (needed for EPR at those checkpoints): about 4 GPU jobs, queue now.

## 4. Eval budget, in priority order

1 eval = 1 detector × 1 checkpoint × 1 dataset, about 3.6 GPU-h. Capacity at about 110/day: about 880 evals by 11/13.

| # | Block | Evals | Done by | Feasible? |
|---|---|---|---|---|
| 1 | Current queue (PTT 27, MSF-4 9, MSF-8 9, CP4F 15, CP-SF 1; unique evals) | about 100 incl. running | 11/6 | **yes** |
| 2 | MSF-4/8 clean predictions e2/e4 | 4 | 11/6 | **yes** |
| 3 | **39 new sets × MSF-4, MSF-8, PTT** (final GT; temporal-fault and pose sets also final no-GT for PTT/MSF-4) | about 140 | 11/8 | **yes** |
| 4 | **MSF-4 no-GT, every checkpoint, injection + removal** (if Delta checkpoints arrive by 11/7) | about 60 | 11/9 | **yes** |
| 5 | **MSF-8 no-GT** (retrain, then evals at e2, maybe e4) | about 10–25 | 11/13 | **uncertain**: only if training starts by about 11/10 |
| 6 | 39 new sets × CP-SF, CP4F (final GT; temporal/pose also no-GT) | about 100 | 11/9 | **yes** |
| 7 | CP4F batch-8 no-GT, every checkpoint, 18 injection sets | about 70 | 11/10–11 | **likely**, if it starts today as estimated (17:12); each day of slip costs a day |
| 8 | CP-SF paste-count 5/30 (2 retrains, then about 20 evals) | 20 | 11/11–12 | **uncertain**: training also faces the GPU queue |
| 9 | Tier C fine-tunes: **MSF/PTT first**, then CP-SF/CP4F (about 40 evals) | 40 | 11/12–13 | **unlikely in full**: needs several training jobs in the same queue; recipe still undecided |
| — | Fidelity-ladder GT effect at every checkpoint | 300+ | — | **no**; use the final-epoch GT − no-GT difference |
| — | Raycast-consistent GT mitigation (CP-SF only) | code + 1 run + 10 | 11/12 | **tight**; only if someone owns the code by 11/7 |
| — | Adversarial-training mitigation; CP-SF all-data runs | — | — | **no** (future work; soften the RQ4 text) |

Evals (rows 1–4, 6) are on track: about 440 evals, about 4–5 days at current throughput. Everything that needs NEW
training (rows 5, 7–9) depends on the GPU queue and is uncertain; plan the paper so it stands without rows 8–9.

**Queue ordering.** Today the pending queue is about 55% PTT, 20% MSF and 25% CP4F (by job count). When the new-set
evals are submitted, MSF/PTT go first; CP-SF/CP4F jobs get a lower priority (`--nice`) so they only fill idle GPUs.

## 5. Coverage estimate

| Scenario | Cells | Of 733 |
|---|---|---|
| Today | 383 | 52% |
| + current queue | about 450 | about 61% |
| + 39 new sets for all five detectors + MSF-4 no-GT (**likely**) | about 580 | about 79% |
| + batch-8 CP4F (likely) | about 600 | about 82% |
| + MSF-8 no-GT at e2/e4, paste-count, part of Tier C (**uncertain**) | about 620–660 | about 85–90% |
| Cut or reframed: fidelity GT effect at every checkpoint, adversarial mitigation, rain/snow, held-out set | — | — |

The temporal-detector rows of every table (GT sampling, temporal faults, history length, lifecycle) are covered at
the 79% level, except MSF-8 no-GT, which is the one gap that depends on getting GPUs for a retrain.
Cell counts beyond the first row are rough (±10%).

## 6. Risks, highest first

1. **No multi-GPU training starts in time** (MSF-8 no-GT, paste-count, Tier C). Fallback: MSF no-GT pair for MSF-4 only,
   MSF-8 GT only; drop paste-count 5/30 and Tier C, or move them to future work.
2. **MSF-4 no-GT checkpoints are not actually finished on Delta**, or transfer is slow. Check today; if not finished,
   retrain on PACE too (about 1.5–2 days on 4 H100, if 4-frame is about half of 8-frame cost — unmeasured).
3. **PACE capacity.** Half of the H100 nodes are down, and our own evals are the largest block in both queues. If the H100
   nodes stay down, eval throughput drops too.
4. **Open paper decisions** (Tier C recipe, GT-effect definition, spray reporting) cost about 1 GPU-day per day of delay.

## 7. Actions

- [x] 39 new sets generating on PACE CPU (11/5).
- [ ] **Today:** confirm MSF-4 no-GT on Delta is finished; start Globus transfer of its checkpoints to PACE (cedar).
- [ ] **Decision pending:** submit MSF-8 no-GT on PACE L40S as a second chance (and hold CP-SF/CP4F evals), or rely on Delta.
- [ ] 11/6: MSF-4/8 clean predictions e2/e4.
- [ ] 11/6: as soon as each new set finishes, submit MSF-4, MSF-8, PTT evals; CP-SF/CP4F behind them with `--nice`.
- [ ] 11/6: decide Tier C recipe and whether paste-count 5/30 is still worth queueing.
- [ ] Ongoing: score with `paper_metrics.py` as evals land.

## 8. Update 11/5 night: main paper first, supplement after submission

The supplement is due **one week after 11/16**, so until 11/14 the GPUs go to main-paper items; supplement-only work
(temporal faults, pose noise, viewpoint phantom, class-wise adaptive sets, history interventions, frame sweep) is
evaluated 11/14–11/23. Main paper = `sec/6_results.tex`; supplement = everything after `\appendix` in `main.tex`.

Every eval that was already queued feeds a main-paper item (fidelity ladder: trace, jitter; Waymo main: removal;
selectivity figure: baselines and Gaussian), so nothing was held.

**Submitted 11/5 (main paper; each starts when its dataset finishes generating, NICE=0):**

| Block | Sets | Models | Jobs (H100/L40S twins) |
|---|---|---|---|
| Fidelity ladder + selectivity | spray_l1, spray_l2, spray_l1_v8, crosstalk_006/010, fog_a002/a006 | GT final: CP-SF, CP4F, PTT, MSF-4, MSF-8; no-GT final: CP-SF, CP4F, PTT | 21 |
| GT-sampling table, removal control (was missing) | atlas_removal_az10–60 | no-GT final: CP-SF, CP4F, PTT (GT final already queued) | 6 |
| Dose-response (class) | attenuated pedestrian (6) + cyclist easy/medium (4) | Tier B: CP-SF and CP4F GT/no-GT at every saved epoch, PTT GT/no-GT e2/4/6 | 55 |

Launchers: `cvpr_scripts/launch_main_new_evals.sh`, `cvpr_scripts/launch_abl_evals.sh` (PACE; not in this repo).
Not yet submitted for the main paper: MSF on the pedestrian/cyclist sets (no MSF no-GT exists, so it adds nothing to the
GT effect), MSF clean predictions e2/e4.

## 9. Training: estimates and what is realistic (main paper)

Timing basis (measured): CP4F no-GT, July, 8 L40S, batch 16: 37.6 h for 36 epochs (about 1 h/epoch). MSF-8 GT, March,
PACE, 2 GPUs, total batch 4: about 21 h/epoch. PACE start estimates on 11/5: 4 × H100 → Nov 20; 4 × L40S → Nov 10;
8 × L40S (CP4F batch-8, already queued) → 11/5 17:12 after three slips.

| Run | Main-paper item | Compute once running | Earliest realistic start | Realistic outcome by 11/13 |
|---|---|---|---|---|
| CP4F no-GT batch 8 (queued) | GT-sampling / factorial control | 8 L40S, about 1.5–2.5 days (batch 8 = twice the iterations of batch 16) | 11/5–6 | **Likely**: done 11/7–8, evals by 11/9–10 |
| MSF-4 no-GT | GT-sampling table, MSF-4 row | already trained on Delta (per Delta, 28 h on 4 A100) | — | **Likely**, if the checkpoints exist and are copied to PACE by 11/7 |
| MSF-8 no-GT | GT-sampling table, MSF-8 row | 4 GPUs, about 10–11 h/epoch → about 2.5–3 days for 6 epochs (assumes linear scaling) | Delta: unknown (stuck); PACE L40S about 11/10 | **Partial at best**: e2 (maybe e4) by 11/13. Fallback: MSF-8 GT only |
| CP-SF paste-count 5 and 30 | dose-response (paste-count) | about 1 day each (estimate; CP-SF trains on every 5th frame) | about 11/9–10 (same queue) | **Uncertain**: one or both might land by 11/12 |
| Tier C fine-tunes (fine-tune without GT sampling) | fade figure, mitigation table | CP4F: about 1 h/epoch on 8 GPUs, so a few epochs is cheap; CP-SF cheaper; MSF about 0.5 day/epoch on 4 GPUs | after the recipe is decided; queue as above | **CP-SF / CP4F likely** if decided by 11/7; **MSF / PTT unlikely** |
| Raycast-consistent GT sampling | mitigation table | new code (1–2 days) + 1 CP-SF run (about 1 day) | about 11/9 if code is ready 11/7 | **Unlikely** without an owner this week |
| Adversarial training (phantoms as negatives) | mitigation table | new code + 2 runs; held-out setting undefined | — | **Not realistic**; future work |

**Order to submit (once decided):** MSF-8 no-GT on PACE L40S (second chance; keep the Delta job), then CP-SF paste-count
5/30, then Tier C CP-SF/CP4F. All share the same congested queue; nothing is submitted yet (decision pending; see
`OPEN_QUESTIONS.md`).
