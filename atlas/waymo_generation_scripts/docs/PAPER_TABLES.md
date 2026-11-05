# Paper table templates (CVPR'27, 50-segment protocol)

Every table is on the same 50 Waymo val segments. Detector set: the 5 paper models (CP-SF, CP4F, MSF-4, MSF-8,
PTT) plus each teammate's Waymo models (PHiM, LION, DSVT, CenterFormer, ...) evaluated with the same protocol.
"GT / no-GT" = trained with / without GT sampling. Unless stated, a cell is ASR % at score >= 0.1 (with the 0.7
cutoff in the appendix) and the final epoch.

| # | Table | Rows | Columns | Datasets (`datasets.sh` names) |
|---|---|---|---|---|
| T1 | Point injection, main | detectors (GT final epoch) | 6 settings: Rel easy/medium/hard, Glo easy/medium/hard; each L2 mAP · ASR | `{rel_fixed,global}_{easy,medium,hard}_att_seed42` (attenuated = realistic attacker); adaptive version in the appendix |
| T2 | Intensity realism | detectors | adaptive / attenuated / trace, per setting (easy, hard) | `<split>_seed42`, `_att_seed42`, `_trace_seed42` |
| T3 | Spoofer jitter | detectors | attenuated / + per-point / + per-frame, per setting | `<split>_att_seed42`, `_att_jpt_50`, `_att_jfr_50` |
| T4 | GT sampling vs no GT sampling | detector x family | ASR per study + GT effect at equal clean recall (OLS over epochs, 95% CI) | every injection dataset |
| T5 | Ablations: object and schedule | detector x family | car global / flicker / random object; cyclist global / relative / flicker / random object; adaptive and attenuated | `abl_car_atlas_hard_*`, `abl_cyc_atlas_*` |
| T6 | Paste (idealized spoof) | detector x family | car / cyclist x global, relative, flicker, random object | `abl_{car,cyc}_paste_*` |
| T7 | Point removal | detectors (GT final epoch) | removal width 10-60 deg; each L2 mAP · removal ASR | `atlas_removal_az{10..60}_seed42` |
| T8 | Removal vs non-adversarial point loss | detectors | ATLAS az20/az50 vs matched-count, uniform 50/90, ring 16/32 | `atlas_removal_az{20,50}`, `baseline_*` |
| F1 | Attack lifetime | onset / decay horizon | ASR per temporal detector, adaptive vs attenuated | main splits (+ per-frame jitter) |
| F2 | Clean accuracy vs ASR | checkpoints (all epochs) | clean vehicle recall vs ASR, GT vs no-GT | T4 data |
| A | Appendix | every table at score cutoffs 0.1/0.3/0.5/0.7, any-class ASR, phantom score distribution, per-segment CIs | | |

Planned additional rows/tables (Delta, clean scenes): road spray, crosstalk, fog (L2 drop + phantom rate),
frustum placement (ASR), blackout / stuck / delayed / dropped-history / pose-noise events (recall over time,
recovery frames); see STUDY_PLAN.md.

Epochs for T4/F2: CP-SF e9/18/27/30, CP4F e9/18/27/36, PTT e2/4/6, MSF-4/8 e2/4/6 (teammates: their own epoch
sweep of a GT and a no-GT run, or final epochs only if no no-GT model exists).
