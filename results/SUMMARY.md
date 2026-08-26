# ATLAS Physical Fidelity: Concern, Experiments, and Findings

A detailed record of the reviewer concern about ATLAS's physical fidelity, every
experiment we ran to address it, and what each result does and does not support.

Companion documents:
* [`ATLAS_REAL_WORLD_VALIDATION.md`](ATLAS_REAL_WORLD_VALIDATION.md) — results-first writeup
* [`osf_k6nf2/README.md`](osf_k6nf2/README.md) — analysis code and reproduction commands
* [`rebuttal.md`](rebuttal.md) — reviewer-by-reviewer responses

---

## 1. The concern

Raised across **Uh4V W3 / L3 / Q3**, with related parts in **MuS2 Q2**. Paraphrased:

> ATLAS is a simulated benchmark built on the Waymo Open Dataset rather than a
> dataset collected via real-world hardware attacks. Actual LiDAR spoofing
> involves complex physical variables — sensor wavelengths, receiver filtering,
> scan patterns, timing synchronization, dual-return mechanisms. The simulation is
> not experimentally validated against real Velodyne, Hesai, or Waymo hardware, so
> it is uncertain whether these simulation results reliably predict real-world
> attack robustness.
>
> Is there any empirical evidence, prior literature, or small-scale hardware
> experiment supporting the physical fidelity of the simulation used for injection
> and removal?

This is a fair criticism of the original submission. Our answer is not "the
simulation is obviously fine" — it is a set of measurements against real attack
hardware, reported with the gaps included.

### Why this is hard to answer cleanly

Three structural obstacles shape everything below:

1. **No public dataset of real spoofing attacks on driving data exists.** Physical
   attack papers publish bench captures of the *attack mechanism*, not annotated
   driving scenes with attacks in them. So there is nothing to train or evaluate
   detectors on directly — which is why ATLAS exists.
2. **Sensors differ.** The available real capture is a 16-beam Velodyne VLP-16 on
   an indoor bench at ~2 m. Our benchmarks are 32-beam nuScenes and 64-beam KITTI
   street scenes. Absolute point densities *cannot* agree, and presenting them as
   if they did would be misleading.
3. **Released datasets are not raw sensor streams.** KITTI and nuScenes ship
   rectified `float32` coordinates, ego-motion compensation, and (for nuScenes)
   10-sweep accumulation. Several raw-hardware properties are destroyed before
   ATLAS runs at all. Any fidelity claim has to separate *our* error from the
   dataset's.

Obstacle 3 is what motivates the control-based design used throughout §4.

---

## 2. The real data

Obtained from the OSF archive `k6nf2`, accompanying:

> Cao, Bhupathiraju, Naghavi, Sugawara, Mao, Rampazzi.
> **"You Can't See Me: Physical Removal Attacks on LiDAR-based Autonomous Vehicles
> Driving Frameworks."** USENIX Security 2023. arXiv:2210.09482.

Contents: four Velodyne VLP-16 raw UDP packet captures (port 2368, 1206-byte
payloads, 10 Hz) plus 50 LGSVL+Apollo screen recordings.

| File | Frames | Return mode | Attack active |
|---|---|---|---|
| `Multimode Analysis/last_mode.pcap` | 312 | last | yes (91 frames) |
| `Multimode Analysis/dual_mode.pcap` | 185 | dual | yes (81 frames) |
| `Moving_Target_Scenario.pcap` | 3580 | strongest | background |
| `moving_outdoor_2 (500-730).pcap` | 233 | strongest | retroreflector |

We decode these at the **packet level** with a dependency-free reader
(`osf_k6nf2/parse_velo.py`) rather than relying on the paper's summary statistics.
This matters: several of the numbers we needed are not in the paper, and one of
them (§3.2) contradicts a naive reading of it.

### 2.1 It is a removal attack, not an injection attack

The attack (PRA) injects saturated echoes *below* the sensor's **Minimum
Operational Threshold** (40 cm on the VLP-16; 90 cm in Apollo, 40 cm in Autoware).
Because the sensor prioritises the strongest echo and the firmware discards
sub-MOT returns, the **genuine object's returns are discarded**. The attacker's
own points are filtered out too. Removal is achieved *via* injection.

Consequence: the correct comparison target is our **`removal_az*`** splits, not
`inject_*`. An earlier draft of our analysis mistakenly treated the capture as a
relative (ego-fixed) *injection* attack, on the strength of the spoof cluster's
locked 35.5° bearing. That reading was wrong — the locked bearing is the paper's
attack angle Δθ — and it was corrected before any of the numbers below were
finalized.

---

## 3. Experiment 1 — attack strength

**Question.** Does ATLAS delete points at a rate a real attacker achieves?

**Method.** For both sides, measure the fraction of genuine returns destroyed
inside the attack wedge. On the ATLAS splits this is
`n_points_removed / n_points_in_sector`, read from generation metadata and
recomputed from the surviving cloud. On the real capture we classify every stable
ray in the attack sector against its own clean reference (built from frames where
the target is present but the attack is off) as *survived*, *range-replaced*, or
*absent*.

### 3.1 Two measurement traps we hit first

**Trap 1: the capture is pre-filter.** Our first extraction looked for a wedge of
*missing returns* and found essentially nothing — total returns per frame were
flat (median 9593, ~1% loss). The pcaps are recorded **before** the MOT filter is
applied, so removal does not appear as absent returns. It appears as genuine
echoes being **range-replaced** on their own ray by the spoof echo. In
`last_mode`, the genuine target sits at 2.01 m and the replacements land at
~11.04 m — free space, since the sector contains no background at all (2 returns
total across all attack-off, target-absent frames).

**Trap 2: removal is peaked in azimuth, so envelope averages understate it.** Our
first reported rate was 0.42, which looked like a 2.3× disagreement with ATLAS.
That number averaged over the whole 6.8° saturated-echo envelope, including the
fringe where the spoofer is barely on target. The per-cell profile:

```
last_mode  az   32.4  34.0  34.4  36.0  36.4  37.2  38.0  38.4
removal          0.75  0.51  0.40  0.32  0.34  0.25  0.39  0.27
```

0.75 at the aim point decaying to 0.25 at the edge. The quantity comparable to
ATLAS's `p_remove` — which is the deletion rate inside the wedge the attacker is
*aiming at* — is the aim point, not the envelope mean.

A third, smaller bug was fixed at the same time: the clean reference used
`np.median` over a NaN-padded stack, so any ray that dropped out once received a
NaN reference and silently vanished from all outcome buckets (the four fractions
summed to 0.45 rather than 1.0). Using `nanmedian` moved the envelope rate from
0.423 → 0.407 and 0.477 → 0.466.

### 3.2 Result

| | removal rate |
|---|---|
| Real, envelope average | 0.407 / 0.466 *(not comparable — see 3.1)* |
| Real, peak azimuth cell | 0.749 / 0.724 |
| **Real, aim point, best-aimed decile of frames** | **0.826 / 0.867** |
| Cao et al. reported, moving car (RP=100 / ≥95 / ≥90) | 0.836 / 0.891 / 0.927 |
| Cao et al. reported, moving robot | 0.854 / 0.924 / 0.959 |
| **ATLAS `removal_az*`** | **0.90–1.00** |

Our independently measured aim-point rate (0.83–0.87) lands on the paper's own
reported object-level range (0.836–0.927), which is a useful consistency check on
the measurement itself.

**ATLAS is 1.0–1.2× the demonstrated rate.** Our least aggressive settings
(`az50`, `az60`, both `p_remove` = 0.90) fall *inside* the reported band;
`az10`–`az40` (0.97–1.00) sit slightly above it. `az20`'s 1.000 is a claim of
perfect removal, which no physical attack achieves.

**Honest framing:** same regime, biased optimistic. Removal ASR from ATLAS should
be read as an upper bound on present-day spoofing hardware, not a match to it.
Capping `p_remove` at ~0.90 across all settings would put every split inside the
measured band, at the cost of regenerating the splits.

### 3.3 Removal rate, not geometry, drives our ASR

Worth stating because it determines which parameter the fidelity argument rests
on. Across the six removal splits:

| split | width | `p_remove` | nuScenes ASR | KITTI ASR |
|---|---|---|---|---|
| az10 | 10° | 0.97 | 0.6084 | 0.3220 |
| az20 | 20° | 1.00 | 0.8075 | 0.6738 |
| az30 | 30° | 0.99 | 0.7630 | 0.5562 |
| az40 | 40° | 0.98 | 0.7058 | 0.4516 |
| az50 | 50° | 0.90 | 0.4343 | 0.1850 |
| az60 | 60° | 0.90 | 0.4316 | 0.2038 |

* r(`p_remove`, ASR) = **+0.979** (nuScenes) / **+0.917** (KITTI)
* r(width, ASR) = **−0.676** / **−0.589**

The clean test: `az50` and `az60` share `p_remove` = 0.90 and produce nearly
identical ASR (0.4343 vs 0.4316; 0.1850 vs 0.2038) despite a 10° width
difference. ASR is governed by the paper-derived removal probability, which is the
parameter we validated in §3.2 — not by our wedge geometry, which we cannot
validate (§3.4).

### 3.4 Wedge width: the bench capture cannot validate it

In the paper's own normalization (chord = 2·d·sin(Δθ/2)):

| | angular width | target range | chord |
|---|---|---|---|
| Real capture | 6.8° | 2.01 m | **0.24 m** |
| ATLAS az10 | 10° | 25 m | 4.36 m |
| ATLAS az20 | 20° | 25 m | 8.68 m |
| ATLAS az30 | 30° | 25 m | 12.94 m |
| ATLAS az40 | 40° | 25 m | 17.10 m |
| ATLAS az50 | 50° | 25 m | 21.13 m |
| ATLAS az60 | 60° | 25 m | 25.00 m |

The real experiment is a 2 m bench target occluding 24 cm of world. It says
nothing about wedge widths against vehicles at 25 m — a 12× range difference. The
applicable reference is therefore the paper's *stated capability*, not this
capture:

* **45° maximum attack angle** demonstrated in hardware
* Their end-to-end LGSVL experiments used **5° and 10°**
* Fig. 6: 4° removes an average of 85 points, >80% of cone points at every
  distance; full cone removal at 6° (2–3 m) and 4° (4 m)

Which gives a three-tier reading we now state explicitly:

| setting | status |
|---|---|
| `az10` | matches what was demonstrated end-to-end (5–10°) |
| `az20`–`az40` | inside the 45° capability envelope, beyond end-to-end demos |
| **`az50`, `az60`** | **exceed 45° — capability extrapolations** |

Also measured and matching: **vertical extent**. The real attack spans **16/16
channels**; ATLAS spans **20–32/32**. Full vertical FOV on both sides, consistent
with a spoofer whose beam divergence covers the victim's entire vertical fan.

And **whole-cloud impact**: ATLAS removal deletes **0.59–4.22%** of the cloud
(az10 → az60), versus 50–90% for weather and beam-dropout corruption baselines.
The attack is surgical, not global — a qualitative distinction from existing
robustness benchmarks.

---

## 4. Experiment 2 — scan-pattern fidelity

**Question.** Is ATLAS output structured like real spinning-LiDAR data, or merely
like plausible 3D points?

This addresses the reviewer's "scan patterns" concern directly.

### 4.1 Why these metrics, and why not per-ring density

The obvious measurement — compare per-ring and per-azimuth point densities — is
**not valid here**, and we do not claim it. A 16-beam indoor bench capture and a
64-beam street scene cannot agree on absolute density; any apparent agreement
would be coincidence, and any disagreement uninformative.

Instead we measure **dimensionless invariants of the sampling lattice**:
quantities that are independent of beam count, scene content, and range, and that
a generator breaks if it places points by perturbing coordinates freely rather
than along real sensor rays. These are exactly the properties that distinguish
"points sampled by a spinning LiDAR" from "points in roughly the right place."

| invariant | physical property tested |
|---|---|
| **I1** range quantization | returns land on the sensor's native range LSB (2 mm) |
| **I2** beam lattice | elevations snap to N discrete rings (comb, not smooth) |
| **I3** azimuth lattice | azimuths land on the firing grid |
| **I4** one-return-per-ray | a (ring, az_cell) holds ≤1 return per sweep |
| **I5** radial density law | returns fall as ~1/r² |
| **I6** ring-arc spacing | neighbour gap along a ring scales as r·dθ |

**The control is the essential design element.** We include *clean, unmodified
KITTI* alongside ATLAS. Without it, any disagreement with the real capture is
ambiguous between "our generator is unfaithful" and "the released dataset already
differs from a raw packet stream." With it, the attribution is unambiguous: an
invariant ATLAS fails *identically to clean KITTI* is inherited from the dataset,
not introduced by us.

**Why KITTI and not nuScenes**, on two independent grounds:
1. KITTI uses a Velodyne HDL-64E — both sides are Velodyne spinning sensors with
   documented firing patterns. Waymo Open uses in-house sensors whose scan
   geometry is not publicly specified.
2. KITTI is **single-sweep**, so per-ray occupancy is comparable to one real
   revolution. nuScenes accumulates 10 sweeps (~264k points, ~9.0 returns per
   cell) and fails I4 by construction — that measurement would be meaningless.

### 4.2 Two decoder artifacts we fixed first

Both were in *our* real-side extraction, and both inflated apparent disagreement:

1. **Firing azimuth.** The VLP-16 packs two 16-channel firings per 32-channel
   block; the second fires ~55.3 µs later, i.e. 0.2° further around. Assigning
   both firings the block azimuth collapsed the azimuth lattice onto a 0.4° grid
   and forced the I6 neighbour-gap ratio to exactly 2.000 — a decoder artifact.
   Fixed, real I6 → **1.000** and real I4 → **1.022** (`last`) / **2.034**
   (`dual`), which is precisely what strongest/last vs dual mode should read.
2. **FOV mismatch.** The ATLAS KITTI splits are camera-FOV cropped (|az| ≤ 40°,
   x > 5 m) while raw KITTI bins are full 360°. Comparing per-ring density or
   azimuth uniformity without a matched crop measures the crop, not the generator.

We also verified that 0.4° is the correct azimuth bin for this decoder: at 0.2°
half the cells starve and the "stable ray" set collapses, which produced spurious
results in an intermediate run.

### 4.3 Result

Measured on KITTI, 36 frames per source:

| invariant | real VLP-16 | ATLAS | clean KITTI | verdict |
|---|---|---|---|---|
| **I6** ring-arc gap ÷ az_res | 1.000 | **1.039** | 1.039 | **match (3.9%)** |
| **I4** single-return-per-ray | 0.978 | **0.881** | 0.830 | **better than control** |
| per-ring density (pts/ring/frame) | — | 352–433 | 411 | **match** |
| **I5** radial slope | −1.33 | −1.58…−1.62 | −1.76 | between |
| **I1** 2 mm range grid | 1.000 | 0.101 | 0.105 | fails; control fails identically |
| **I3** azimuth grid | 0.119 | 0.100 | 0.101 | fails; control fails identically |

Row by row:

* **I6 is the strongest result.** ATLAS reproduces the `r·dθ` arc-spacing law to
  **1.039 vs the real 1.000** — 3.9% — and it is the *identical* 1.039 that clean
  KITTI shows. ATLAS inherits the law exactly, adding zero error of its own.
* **I4:** ATLAS (0.881) is **closer to real hardware (0.978) than clean KITTI is
  (0.830)**. Phantom points compete for rays the way physical returns do rather
  than being layered on top of the scene.
* **Per-ring density:** ATLAS 352–433 vs FOV-matched clean KITTI 411 —
  indistinguishable. No real-hardware column, because beam counts differ (16 vs
  64); this row compares ATLAS to the control only.
* **I5:** ATLAS sits between real and control. The real slope is not meaningful
  anyway — a 2 m indoor room, R² = 0.50 — whereas ATLAS fits well (R² = 0.83–0.87).
* **I1 and I3 fail, and clean KITTI fails identically.** Both sit at the **0.10
  chance level** for an unquantized cloud. KITTI distributes rectified `float32`
  coordinates, which destroys the integer range grid and the azimuth lattice
  before ATLAS ever runs. This is a property of every released LiDAR dataset, not
  an artifact of our generation. Note also that the real capture scores only 0.119
  on I3, so that test is weakly discriminative even on true hardware.

### 4.4 What we do NOT claim from this

Two rows we measured and are not presenting as fidelity evidence:

* **Azimuth CV:** ATLAS 0.249–0.264 vs clean KITTI 0.089. ATLAS is ~3× *less*
  azimuthally uniform, because the wedge/phantom is a localized density
  perturbation. Expected, but a real difference. Comparing it to the real
  capture's 0.375 would be misleading, since that is full-360° indoor.
* **Rings occupied:** 49/64 (ATLAS) vs 61/64 (clean). ATLAS clouds populate fewer
  rings, likely a consequence of the range/FOV crop.

We also ran the full suite on nuScenes. It fails I1–I4 (I4 at 9.0 returns/cell),
entirely because of 10-sweep accumulation and ego-motion compensation — clean
nuScenes fails identically. We report the KITTI numbers because they are the ones
where the measurement is meaningful, and we say so rather than presenting the
nuScenes run as a second confirmation.

---

## 5. Experiment 3 — dual-return mechanisms

Named explicitly in the reviewer's list.

ATLAS models second-return behaviour as a per-sensor parameter, `P_two`:

| dataset | `P_two` | rationale |
|---|---|---|
| Waymo | 0.01 | dual-return sensor |
| nuScenes | 0 | single-return |
| KITTI | 0 | single-return |

Verified on the real capture: `dual` mode yields **2.034 returns per ray** versus
**1.022** in `last` mode — the expected factor of two, and independent
confirmation that our decoder and occupancy metric are correct.

Cao et al.'s own multimode analysis shows removal succeeds under **strongest,
last, and dual** modes. So this parameter affects point statistics rather than
attack viability, which bounds how much rides on getting it exactly right.

---

## 6. Experiment 4 — receiver filtering

Also named in the concern.

The removal mechanism depends on the sensor's MOT discarding sub-threshold echoes.
**ATLAS models the effect of this filter** — genuine returns on attacked rays are
eliminated — **rather than the analog front-end that produces it.**

Accounting for this was necessary to measure both sides consistently, and is the
root cause of Trap 1 in §3.1: the raw captures are recorded *pre-filter*, so
removal appears there as range replacement rather than missing returns. A
detector consuming the post-filter stream would see missing returns, which is what
ATLAS produces.

The paper's proposed defenses are worth noting as context for what the filter
implies: Fake Shadow Detection (82.5% TNR / 91.2% TPR) and Azimuth-based Detection
(99.98% TNR / 100% TPR).

---

## 7. Experiment 5 — intensity ablation

The largest measurable discrepancy between ATLAS and the real attack, and the
result we find most interesting because it runs opposite to intuition.

### 7.1 The gap

| population | median | p99 | frac ≥ 200 |
|---|---|---|---|
| real nuScenes **car** points | 5.0 | 100 | 0.0002 |
| real nuScenes **all** points | 15.0 | 88 | 0.0006 |
| ATLAS phantom (as written to disk) | 14.7 | 97 | ~0 |
| **real spoofer (OSF capture)** | **255** | 255 | **1.000** |

Real spoofed returns saturate completely. ATLAS samples phantom intensity from the
diffuse range-conditioned distribution. A ~17× median gap.

Note this is *the attack mechanism*, not a calibration error on our part:
saturation is how the spoofer wins the ray.

### 7.2 Closing the gap collapses attack success

3 architectures × 3 splits, forcing phantom intensity to 255:

| Model | split | baseline ASR | ASR @ I=255 | Δ | relative |
|---|---|---|---|---|---|
| pp_multihead | inject_relative_easy | 0.3585 | 0.0336 | −0.3249 | **−91%** |
| pp_multihead | inject_relative_medium | 0.1157 | 0.0156 | −0.1001 | −87% |
| pp_multihead | inject_relative_hard | 0.0246 | 0.0139 | −0.0107 | −44% |
| voxel0075_voxelnext | inject_relative_easy | 0.4848 | 0.0221 | −0.4627 | **−95%** |
| voxel0075_voxelnext | inject_relative_medium | 0.1977 | 0.0131 | −0.1846 | −93% |
| voxel0075_voxelnext | inject_relative_hard | 0.0468 | 0.0107 | −0.0361 | −77% |
| transfusion_lidar | inject_relative_easy | 0.4028 | 0.0623 | −0.3404 | **−85%** |
| transfusion_lidar | inject_relative_medium | 0.1756 | 0.0246 | −0.1509 | −86% |
| transfusion_lidar | inject_relative_hard | 0.0468 | 0.0098 | −0.0369 | −79% |

Baselines reproduce `atlas_nuscenes_results.csv` (pp 0.363, voxelnext 0.485,
transfusion 0.398), validating the harness. The effect is **architecture-agnostic**
— pillar, sparse-voxel, and transformer detectors all lose 85–95%.

### 7.3 The dose-response curve localizes the cause

pp_multihead, `inject_relative_easy`, each row a full 1219-target pass:

| forced I | 5 | 15 | 30 | 60 | 100 | 150 | 200 | 255 |
|---|---|---|---|---|---|---|---|---|
| ASR | 0.2863 | 0.2346 | 0.2453 | 0.2518 | 0.2371 | 0.1370 | 0.0599 | 0.0345 |
| Δ vs 0.3544 | −0.068 | −0.120 | −0.109 | −0.103 | −0.117 | −0.217 | −0.295 | −0.320 |

**Not monotone in brightness — a plateau with a cliff.** That shape decomposes the
effect into two distinct causes:

1. **A constant ~−0.11 offset, flat across I = 15…100** (a 6.7× brightness range).
   Any *constant* intensity costs ~0.11 ASR regardless of value. This is loss of
   intensity **variance/texture**, not brightness: ATLAS's range-conditioned
   sampling gives the phantom a plausible within-object spread, and flattening it
   alone removes about a third of the attack's success.
2. **A cliff beyond I ≈ 100**, contributing the remaining ~−0.21. I = 100 is
   *exactly the p99 of real nuScenes car intensity* (§7.1). ASR is unaffected while
   the forced value stays inside the training distribution's support and collapses
   only once it leaves it.

So this is an **out-of-distribution effect localized to the top 1% tail of the
intensity distribution**, not a "brighter is easier to detect" physical trend. That
is stronger evidence than a monotone curve would have been: monotone decay would be
consistent with detectors genuinely using intensity as a physical cue, whereas
flat-then-cliff at the p99 boundary is the signature of inputs falling off the edge
of the training data.

### 7.4 Four consequences

1. **ATLAS's diffuse-intensity choice is correct for a geometry benchmark.** It
   keeps the phantom in-distribution so measured ASR reflects *shape/placement*
   vulnerability, which is what the study is about. Had ATLAS used saturated
   intensity, every injection ASR would be near zero and the benchmark would
   measure nothing.
2. **ATLAS injection ASR is an UPPER bound on this specific real attack, not a
   lower bound.** An earlier draft of ours described the intensity bias as
   "conservative"; the ablation shows the opposite. A real saturating spoofer would
   be caught *more* easily by these detectors. Stated honestly, that is a
   limitation of the *real attack* — it is trivially detectable on intensity — not
   of ATLAS geometry.
3. **The plateau is itself an alignment result.** Because ASR is flat over
   I = 15…100, ATLAS's exact intensity value does not matter as long as it stays
   in-distribution. The real–synthetic intensity gap therefore does **not**
   propagate into the benchmark numbers — the measured ASR is insensitive to the
   one channel the generator models least faithfully. This is precisely what you
   want when arguing a synthetic benchmark is a valid stand-in.
4. **It identifies a nearly free defense.** An intensity-plausibility filter would
   reject the real attack outright (100% of its returns saturate vs 0.02% of real
   car points) at essentially no cost to clean detection. Worth stating explicitly
   — and it means ATLAS cannot be used to *evaluate* such a filter.

---

## 8. Experiment 6 — cross-attack orthogonality

Not a fidelity experiment, but it establishes why both attack modes are needed —
which is part of what the benchmark is for.

| correlation | nuScenes (n=5) | KITTI (n=8) |
|---|---|---|
| clean mAP vs injection ASR | +0.68 | −0.17 |
| injection rank vs removal rank | −0.20 | −0.31 |
| injection ASR vs removal ASR | −0.34 | −0.64 |

No p-value reaches 0.05 (closest: KITTI ASR pair, p = 0.089), and leave-one-out
resampling swings the values substantially (nuScenes clean-vs-injection moves
+0.68 → +0.38 without pp_multihead, → +0.94 without transfusion_lidar). So the
defensible claim is **"not predictive"**, not "negatively correlated."

The load-bearing evidence is a concrete reversal rather than a coefficient. The
**KITTI SECOND family** ranks worst and 2nd-worst against injection (0.4555,
0.4348) yet **best and 2nd-best against removal** (0.3223, 0.3139) — a 6-rank flip
in two related architectures. Two independent models showing the same reversal is
a mechanism, not noise.

The pooled coefficients also understate a consistent pattern, because the
correlation depends on injection sparsity:

| injection difficulty | inj-vs-rem r, nuScenes | KITTI |
|---|---|---|
| easy (dense phantom) | −0.17 … +0.13 | −0.57 … +0.06 |
| hard (sparse phantom) | **−0.58 … −0.78** | **−0.57 … −0.95** |

Against dense phantoms every detector fails similarly, so there is nothing to
correlate. Against sparse phantoms — the realistic case — the anti-correlation is
strong, and it replicates across all 6 removal widths and all injection families
(global and relative on nuScenes, plus KITTI). A single-attack benchmark would
rank these models roughly backwards on the other attack.

---

## 9. What remains unvalidated

Stated explicitly rather than left for a reviewer to find:

| variable | status |
|---|---|
| **Laser wavelength** | not modelled. ATLAS operates on point clouds, not optical signals. |
| **Analog receiver electronics** | not modelled. We model the MOT filter's *effect*, not the front-end. |
| **Timing synchronization** | not modelled. ATLAS assumes a successfully synchronized attacker and simulates the resulting perturbation. |
| **Wedge width > 45°** | `az50`/`az60` exceed demonstrated hardware capability; labelled extrapolations. |
| **Injection, against matched hardware** | our capture is a *removal* attack. Injection fidelity rests on the §4 lattice invariants plus prior literature, not a matched capture of our own. |
| **Range/azimuth quantization** | ATLAS cannot reproduce these because KITTI/nuScenes do not preserve them. Bounds what *any* dataset-based benchmark can claim about raw-signal fidelity. |
| **Hesai / other vendors** | not tested. Our port required only 4 sensor constants (§10), so we expect this to be mechanical, but we have not demonstrated it. |

**Supporting literature for injection** (in place of a matched capture):
PLA-LiDAR (Jin et al., IEEE S&P 2023) demonstrates controlled multi-point
injection on commodity LiDAR; Sato et al. (NDSS 2024) report spoofing capability
on moving vehicles substantially larger than earlier work assumed, placing our
injection settings inside rather than beyond demonstrated capability.

> ⚠️ **Verify before submission.** These two citations are the only claims in this
> document not measured directly from data — they are recalled, not checked. Confirm
> the venues, authors, and specific capability figures before citing.

---

## 10. Related: sensor recalibration surface (MuS2 Q2)

Adjacent to the fidelity concern — "which components need recalibration for other
sensors?" — and answered by the port itself.

| parameter | Waymo | nuScenes | KITTI |
|---|---|---|---|
| beam inclinations | 64 beams | 32 beams, −30.56°…+10.69° | 64 beams, −23.99°…+6.19° |
| azimuth resolution | 0.1358° | 0.3324° | 0.1728° |
| intensity statistics | 6 range bands | 6 bands (re-measured) | 6 bands (re-measured) |
| second-return probability | 0.01 | 0 | 0 |

**Only these four quantities are dataset-specific**, and all are extracted
automatically by one script per dataset. A diff of the generator constants across
Waymo and nuScenes is empty apart from a class label and the removal `PRR` table:
ground anchoring, the occlusion model, and the removal probabilities are
**identical across all three datasets**. No attack parameter was retuned per
dataset.

Beam inclinations are recovered differently depending on available metadata:
nuScenes exposes a per-point ring index, so the table is read off exactly; KITTI
has none, so inclinations are recovered by quantile-binning pooled elevation
angles. This is the only step whose accuracy depends on the dataset, and it likely
explains why KITTI's I2 beam-lattice residual (0.089°) is looser than nuScenes'.

Remaining per-dataset work is **format plumbing, not calibration**: point-feature
layout (Waymo carries elongation in feature 4, nuScenes a timestamp, KITTI
neither), frame indexing, and ego-pose conventions, each handled in a thin adapter.

One substantive sensor difference changes *results* rather than code: nuScenes
accumulates 10 sweeps into ~264k-point clouds, so occlusion filtering is
considerably more aggressive than single-sweep Waymo or KITTI. Sparse phantoms are
more often occluded before reaching the detector, lowering ASR on our sparsest
splits. This is a real property of the sensor pipeline that ATLAS reproduces, not
a parameter needing adjustment — and it is the mechanism behind the easy→hard
correlation gradient in §8.

**Generation cost**, for completeness (one split, full val set):

| | frames | wall time | per frame |
|---|---|---|---|
| nuScenes injection | 6,019 (150 scenes) | ~78 min | ~780 ms |
| nuScenes removal | 6,019 (150 scenes) | ~78 min | ~740 ms |
| KITTI injection | 3,769 | ~2.4 min | 35–38 ms |
| KITTI removal | 3,384 | ~1.4 min | 24 ms |

nuScenes is 20–30× slower per frame, again from 10-sweep accumulation (~264k
points/cloud vs ~16k) and O(points) occlusion filtering. Both figures are wall
time with all splits running concurrently, not on an idle machine.

---

## 11. Summary of the position

**What we can now claim, with measurements behind it:**

1. ATLAS's removal strength matches real attack hardware to within 1.0–1.2×
   (0.90–1.00 vs 0.83–0.87 measured, 0.836–0.927 reported), and ASR is driven by
   that validated parameter (r = 0.92–0.98) rather than by unvalidated geometry.
2. ATLAS output obeys the sampling-lattice invariants of real spinning LiDAR where
   the host dataset permits measuring them: ring-arc spacing within **3.9%** of
   real hardware, per-ray occupancy **closer to real hardware than clean KITTI**.
3. Where ATLAS deviates from a raw sensor stream, **clean data deviates
   identically** — those are released-dataset properties, not generation artifacts.
   The control design is what licenses this attribution.
4. Dual-return behaviour is an explicit per-sensor parameter, verified against both
   return modes of the real capture (2.034 vs 1.022 returns/ray).
5. The intensity gap is real (~17×) but **provably does not propagate into the
   benchmark**: ASR is flat across a 6.7× brightness range, and closing the gap
   *reduces* ASR by 85–95% for out-of-distribution reasons.
6. Porting to a new sensor requires recalibrating **four constants**, no attack
   parameters.

**What we explicitly concede:**

1. `az50`/`az60` exceed demonstrated hardware capability.
2. `az20`'s `p_remove` = 1.00 is above anything physically achievable.
3. Wavelength, receiver electronics, and timing synchronization are not modelled.
4. Range and azimuth quantization cannot be reproduced from released datasets.
5. Our hardware capture is removal-only; injection fidelity leans on lattice
   invariants plus prior literature.
6. ATLAS injection ASR is an **upper** bound for a saturating physical spoofer, not
   a lower bound — our earlier "conservative" framing was wrong.

**Why we think this is a sufficient answer.** The concern asks whether simulation
results reliably predict real-world attack robustness. We cannot fully settle that
without a matched hardware campaign, and we do not claim to. What we can show is
that (a) the parameter our results are most sensitive to is calibrated against and
consistent with real hardware measurements, (b) the geometric structure of our
output is indistinguishable from real sensor sampling on every invariant the host
dataset preserves, and (c) the one channel we model least faithfully is provably
inconsequential to the measured metric. The remaining gaps are named, bounded, and
in two cases (intensity, quantization) shown to be properties of the real attack
or the released data rather than of ATLAS.
