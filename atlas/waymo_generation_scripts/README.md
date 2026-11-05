# ATLAS Waymo datasets: generation + evaluation kit (CVPR'27, 50-segment protocol)

Start here, then read in this order (an agent should read all of them before acting):
1. `docs/STUDY_PLAN.md` and `docs/EVAL_STATUS.md` (which model / epoch / GT or no-GT is done, running or queued per dataset): what the study is, every dataset and its status, protocol decisions, caveats, planned ablations.
1b. `docs/DATASETS.md`: every one of the 63 datasets, one row each, with a detailed description.
2. this README: setup, banks, generating or subsetting a dataset, the dataset registry.
3. `docs/GENERATION.md`: exactly how each attack is built (schedule, placement, raycasting, intensity, jitter,
   paste, removal, baselines, banks) and the most efficient way to produce a dataset.
4. `docs/MODELS.md`: how each of the 5 detectors was trained (GT vs no-GT recipes, second-stage proposals).
5. `docs/PAPER_TABLES.md`: the paper's table and figure templates and which datasets feed each.
6. `docs/EVALUATION.md`: the five paper detectors, their checkpoints and GT/no-GT families, each inference
   pipeline (`eval/`), how to add your own detector, injection vs removal ASR and every other metric (`eval/scoring/`).

Folder map: `generators/` dataset generators - `banks/` bank builders - `data/` 50 segment names, the paper's
trace libraries, beam inclinations, intensity statistics - `datasets.sh` registry - `generate.sh` driver -
`subset_50.py` reuse of 195-segment splits - `eval/` the paper's eval scripts for CP-SF, CP4F, PTT, MSF-4/8 -
`eval/scoring/` ASR / removal / lifetime / clean-FP / per-frame scorers - `eval/launchers/` PACE launchers (reference).

Scripts that generate every ATLAS point-injection / point-removal dataset used in the CVPR'27 submission, so you
can (1) evaluate your own Waymo detectors on exactly the same data, and (2) port each study to nuScenes / KITTI.
**No data is included**: banks and datasets are built from your own copy of the Waymo Open Dataset.

Protocol shared by every dataset:
- the **same 50 Waymo validation segments** (`data/segments_50_seed42.txt`; the generators draw them with
  `--num_sequences 50 --sequence_selection random --seed 42` from the 195 segments common to the 1- and 4-frame
  OpenPCDet loaders), **seed 42**;
- alternating **32-frame clean / attack blocks** per segment (clean 0-31, attack 32-63, clean 64-95, ...);
- generation is CPU only (OpenPCDet data loaders + numpy raycasting; no model is run).

The generators are the CVPR versions: tanh intensity fix (NeurIPS-era datasets drew spoof intensities on the
raw Waymo scale and must not be reused), intensity modes, spoofer jitter, paired removal, `--data_path`.

---------------------------------------------------------------------------------------------------------------

## Setup

1. Waymo v1.x processed by OpenPCDet (`waymo_processed_data_v0_5_0/`, `ImageSets/`, per-segment infos pkls).
2. An env with OpenPCDet (`pcdet` importable), numpy, torch, matplotlib, plotly (tensorflow +
   `waymo_open_dataset` only for `banks/extract_waymo_inclinations.py`).
3. Edit the paths at the top of `config.sh` (or export `WAYMO_ROOT OPENPCDET OUT_ROOT TRACE_DIR BANK_DIR`).

## Step 0: build the banks (once)

| Bank | Used by | Build |
|---|---|---|
| `traces_{easy,medium,hard}.pkl` (300 real Vehicle point sets each; 100-200 / 300-500 / 500-1000 points) and `traces_val_cyclist_hard.pkl` | every ATLAS car / cyclist injection | **shipped in `data/` (the paper's original pickles; nothing to build)**. `banks/build_trace_banks.py` rebuilds the car libraries for reference (same traces and point sets; row order can differ) |
| `vehicle/object_bank_val.pkl` (1.8 GB), `cyclist/object_bank_val.pkl` (36 MB): real objects from val segments NOT in the 50, with segment/frame/range | the 8 paste ablations only | build: `bash banks/build_val_object_bank.sh` (CPU, 8 cores, 128 GB, a few hours) -> `$BANK_DIR` |
| `data/waymo_top_lidar_inclinations.npy` (64 TOP-LiDAR beam inclinations) | raycasting, ring baselines | shipped; `banks/extract_waymo_inclinations.py` reproduces it exactly from the first val segment |
| `data/range_conditioned_stats.npy` (per range band: lognormal raw-intensity fit, clip range, elongation pool, second-return rate) | adaptive intensity mode only | shipped (the paper's file); `banks/build_range_stats.py` is a documented re-fit for new data, not a bit-exact reproduction |

On PACE you can point `BANK_DIR` at the existing banks instead of building them.

## Step 1: generate

```bash
bash generate.sh --list                       # every registered dataset name
bash generate.sh global_hard_att_seed42       # one dataset -> $OUT_ROOT/global_hard_att_seed42/{dataset,annos,metrics}
```
Reruns resume (finished segments are skipped); each run ends with `verify_dataset.py` (50 segments, annos,
point format, spoof intensities <= 1). Generate the baselines before `atlas_removal_*` (removal checks its
frame/target pairing against `baseline_matched_az50_seed42`). The Gaussian-noise control is a separate script
(`generators/generate_gaussian_control.sh`, edit its EDIT block; it copies the paired ATLAS run's spoof boxes).

Cost per 50-segment dataset (8 cores, measured): global-placement injection ~2 h (peak RAM ~56 GB), relative
~3.3 h (~132 GB), removal / baselines ~1.3 h (16 cores). Output ~40 GB per dataset.

## Step 2 (alternative): pull the 50 segments from a dataset you already have

If you already generated a 195-segment version **with these CVPR generators and the same settings** (e.g. the
attenuated 6 x 195 `<split>_full_att_seed42`), do not regenerate:
```bash
python subset_50.py /path/global_hard_full_att_seed42 $OUT_ROOT/global_hard_att_seed42 [--link]
```
Every random draw is keyed by the seed and the segment/frame, not by how many segments were generated, so the
50-segment subset of a 195-segment split equals the 50-segment generation (spot-checked frame by frame: same
attacked frames and spoof boxes). Not valid for NeurIPS-era datasets (intensity bug) or different settings.

## Already have some data? Decide per dataset before generating anything

Check what you have with `python verify_dataset.py <dir> --expect <N>` and (if present) `<dir>/metrics/run_manifest.json`
(generator name, seed, arguments, intensity mode).

| What you have | Do |
|---|---|
| The 50-segment dataset, complete, made with these CVPR generators and the same settings | nothing; verify and evaluate |
| The same, but incomplete (job died, some segments missing) | rerun `bash generate.sh <name>` with the same `OUT_ROOT`: finished segments are skipped, only the missing ones are generated |
| A 195-segment split made with these CVPR generators and the same settings (e.g. attenuated 6 x 195 `<split>_full_att_seed42`) | `subset_50.py <195 dir> <50 dir>` (copies the 50 segments; `--link` on the same filesystem) |
| A 195-segment split that is only partly generated | `subset_50.py` lists the missing segments; generate the missing ones into the 195 folder first (rerun its 195-segment generation, which resumes), or generate the 50-segment dataset directly with `generate.sh` |
| A dataset on a different segment subset (e.g. the old 30-segment schedule/jitter sets) | not reusable as is: the 50 segments overlap only partly. Use `subset_50.py` only if it reports all 50 present; otherwise regenerate |
| Any NeurIPS-era injection dataset (made before the tanh intensity fix; NeurIPS `generate_data_*` scripts) | regenerate: spoof intensities are on the wrong scale. Exception: paste datasets (real points, no intensity sampling) and removal/baseline datasets made by the rebuttal-era removal/baseline generators with seed 42 are unaffected |
| The same attack but a different intensity mode or jitter setting (e.g. adaptive when you need attenuated) | regenerate with the right name; the geometry is identical across modes (same random stream), so the two are frame-paired, but the files differ |
| Deleted / superseded studies (calibrated, saturated, attenuated v0, combined jitter J1/J2/J3, unpaired removal_20deg/50deg) | not part of the study; ignore |
| Predictions already computed by an eval on a dataset you now replace | discard them; predictions are only valid for the exact dataset files they were run on |

Never mix segments from two different generations into one dataset folder; `subset_50.py` and the resume logic
only add files that belong to the same generation.

---------------------------------------------------------------------------------------------------------------

## Dataset registry (`datasets.sh`)

Splits: `global` = spoof fixed in the world (the ego drives past it); `rel_fixed` = spoof fixed relative to the
ego (15-20 m ahead, +-3 m lateral, moves with it). Budgets: easy 100-200, medium 300-500, hard 500-1000 trace
points before raycasting. Raycasting: 0.1358 deg azimuth bins x 64 beams, spoof points behind a real return are
dropped, real returns behind a spoof point are removed, <= 2 returns per ray; the spoof is re-grounded every frame.

**Intensity modes** (only the spoof points; geometry identical across modes):
`adaptive` = per point, drawn from the range band's raw-intensity fit, then tanh; `attenuated` = constant 0.3221
(tanh scale; q90 of real-car trace intensity, Sako et al. NDSS'24); `trace` = the trace's recorded values.

| Group | Dataset names (all end in `_seed42`) | What it is |
|---|---|---|
| Main injection (12) | `{global,rel_fixed}_{easy,medium,hard}` and `..._att` | ATLAS car spoof, 6 placement x budget settings, adaptive or attenuated intensity |
| Trace intensity (4) | `{global,rel_fixed}_{easy,hard}_trace` | same, spoof keeps the trace's recorded intensities (replay attacker) |
| Spoofer jitter (12, attenuated) | `<split>_att_jpt_50` | per-point jitter: each spoof point moved along its ray by N(0, 0.10 m) (Sato NDSS'24 inner-frame error) |
| | `<split>_att_jfr_50` | per-frame jitter: the whole object shifted along the line of sight by one N(0, 0.35 m) per frame (Sato inter-frame error); scored against the realized box (`spoof_gt`; aimed box in `spoof_gt_intended`) |
| Car ablations (6) | `abl_car_atlas_hard_{global,flicker,randobj}` and `..._att` | hard car trace, world-fixed; `flicker` = spoofer fires each attack frame with p = 0.5; `randobj` = a different random trace every frame |
| Cyclist ablations (8) | `abl_cyc_atlas_{global,relative,flicker,randobj}` and `..._att` | cyclist trace bank, target class Cyclist (IoU 0.5) |
| Paste (8) | `abl_{car,cyc}_paste_{global,relative,flicker,randobj}` | most idealized spoof: a real object's points and intensities from the val object bank (recorded at the same range, rotated to the bearing), all points kept, scene points in the box removed (as OpenPCDet GT sampling) |
| Removal (6) | `atlas_removal_az{10,20,30,40,50,60}` | ATLAS A-HFR point removal: a tracked, servo-limited sector (width W deg x 16 deg) around one target vehicle is emptied with removal probability 0.97/1.0/0.99/0.98/0.90/0.90; all widths attack the same frames and targets |
| Point-loss baselines (6) | `baseline_uniform_{50,90}`, `baseline_ring_{16,32}`, `baseline_matched_az{20,50}` | scene-wide, non-adversarial, same frames/targets as removal: random point dropout p; drop N of 64 elevation layers; delete as many points as ATLAS would, uniformly from the whole scene |
| Control (1) | `gaussian_sigma_008_global_easy` (separate script) | Gaussian xyz noise sigma 0.08 m on the attack frames of global_easy instead of a spoof (expected: no phantoms) |

## Evaluating

- **Injection ASR**: per attacked frame, among predictions with score >= t, take the one with the highest 3D IoU
  to `spoof_gt`; success if it is the target class at IoU >= 0.7 (Vehicle) / 0.5 (Ped, Cyclist). Report
  t in {0.1, 0.3, 0.5, 0.7}; ASR = successes / attacked frames (frames whose spoof lost all points count as failures).
- **Removal ASR** (different rule): target frames = annos records with `active_this_frame`; the target (`spoof_gt`,
  class = last element) is *detected* if, among predictions **of the target class**, the best 3D IoU >= 0.7;
  ASR = hidden / active frames.
- Waymo L2 mAP/mAPH over all frames of the 50 segments. Paired segment bootstrap for CIs.
- Multi-frame detectors must rebuild their history from the stored corrupted single-frame clouds
  (`dataset/<segment>.pkl`, ego poses in each frame dict), not from the clean loader.

## Porting to nuScenes / KITTI

Keep the protocol (seeded segment/scene subset, 32-frame blocks, the same placement draws) and replace the
sensor-specific pieces: beam inclinations and azimuth resolution (raycasting), the intensity scale and the
attenuated constant (recompute q90 of real-car points on that dataset's loader scale), the range bands of
`range_conditioned_stats`, and the trace / object banks (build them from that dataset's val split, excluding
the attacked scenes). `atlas/` already holds NeurIPS-era nuScenes/KITTI ports of the global/relative/removal
generators and asset builders; apply the CVPR changes listed at the top of this file to them.
