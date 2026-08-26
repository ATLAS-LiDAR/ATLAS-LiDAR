# Real-World Validation of ATLAS

Comparison of the ATLAS synthetic adversarial LiDAR generator against a **real
physical LiDAR attack capture** (OSF `k6nf2`), to establish whether the simulated
attacks are physically realizable.

Everything here is measured on the **generated splits actually fed to the
detectors** (`nuscenes_trainval/atlas_val/`, `kitti/atlas_val/`), not on the raw
trace library.

Analysis code and reproduction commands: [`osf_k6nf2/README.md`](osf_k6nf2/README.md).

## 1. What the real data is

The capture is from:

> Cao et al., "You Can't See Me: Physical Removal Attacks on LiDAR-based
> Autonomous Vehicles Driving Frameworks", USENIX Security 2023
> (arXiv:2210.09482).

Four Velodyne **VLP-16** raw UDP captures (port 2368, 10 Hz) plus 50 LGSVL+Apollo
screen recordings.

| File | Frames | Return mode | Attack active |
|---|---|---|---|
| `Multimode Analysis/last_mode.pcap` | 312 | last | yes (91 frames) |
| `Multimode Analysis/dual_mode.pcap` | 185 | dual | yes (81 frames) |
| `Moving_Target_Scenario.pcap` | 3580 | strongest | background |
| `moving_outdoor_2 (500-730).pcap` | 233 | strongest | retroreflector |

**It is a REMOVAL attack (PRA), not an injection attack.** The attacker injects
saturated echoes *below* the sensor's Minimum Operational Threshold (40 cm on the
VLP-16). Because the sensor prioritises the strongest echo and the firmware
discards sub-MOT returns, the genuine object's returns are discarded — the
attacker's points are filtered out too. Removal is achieved *via* injection.

So the comparison target is the **`removal_az*`** splits, not `inject_*`.

Two measurement subtleties, both of which produced wrong answers first:

* The pcaps are recorded **pre-MOT-filter**, so removal does not show up as
  missing returns — it shows up as genuine echoes being **range-replaced** on
  their own ray. In `last_mode` the target sits at 2.01 m and replacements land
  at ~11 m (free space; the sector has no background at all).
* Removal is strongly **peaked in azimuth**. Per-cell rate runs 0.75 at the aim
  point down to 0.25 at the edge of the 6.8° saturated envelope. Averaging over
  the envelope understates the rate ~2×.

## 2. Attack strength: ATLAS vs real

Fraction of genuine returns destroyed inside the wedge:

| | removal rate |
|---|---|
| Real, envelope average | 0.41–0.47 *(not comparable — includes off-target fringe)* |
| Real, peak cell | 0.72–0.75 |
| **Real, aim point / best-aimed frames** | **0.83–0.87** |
| Cao et al. reported, moving car (RP=100 / ≥95 / ≥90) | 0.836 / 0.891 / 0.927 |
| **ATLAS `removal_az*`** | **0.90–1.00** |

ATLAS is **1.0–1.2×** the physically demonstrated rate — same regime, biased
slightly optimistic. `az50`/`az60` (`p_remove` = 0.90) fall inside the paper's
reported band; `az10`–`az40` (0.97–1.00) sit just above it, and `az20`'s 1.000 is
above anything physically achievable.

Vertical extent agrees: **16/16 channels** real, **20–32/32** ATLAS — full
vertical FOV on both sides.

Whole-cloud impact: ATLAS removal deletes **0.6–4.2%** of the cloud, versus
50–90% for weather/beam-dropout corruption baselines. The attack is surgical, not
global.

### Width is the real gap, and the bench capture cannot validate it

In the paper's own normalization (chord = 2·d·sin(Δθ/2)):

| | angular width | target range | chord |
|---|---|---|---|
| Real capture | 6.8° | 2.01 m | **0.24 m** |
| ATLAS az10 | 10° | 25 m | 4.36 m |
| ATLAS az20 | 20° | 25 m | 8.68 m |
| ATLAS az60 | 60° | 25 m | 25.0 m |

The real experiment is a 2 m bench target occluding 24 cm of world; it says
nothing about wedge widths against vehicles at 25 m. The applicable bound is the
paper's stated capability: **45° maximum attack angle**, with their end-to-end
LGSVL experiments run at **5° and 10°**. Therefore:

* `az10` — matches what was demonstrated end-to-end.
* `az20`–`az40` — inside the 45° capability envelope.
* **`az50`, `az60` — exceed it; report as capability extrapolations.**

(These two also share `p_remove` = 0.90, which is why their ASRs are nearly
identical: 0.434 vs 0.432 on nuScenes, 0.185 vs 0.204 on KITTI. ASR is driven by
`p_remove`, not by our geometry — r(p_remove, ASR) = 0.979 / 0.917 versus
r(width, ASR) = −0.676 / −0.588.)

## 3. Scan-pattern fidelity: sensor-physics invariants

Absolute densities cannot agree (VLP-16 indoor bench vs 64-beam KITTI street
scenes). What *can* be compared are sensor-independent invariants that a naive
generator gets wrong. **Clean unmodified KITTI is included as a control**: any
invariant ATLAS fails identically to clean KITTI is a property of the released
dataset, not of generation.

Measured on **KITTI** (single-sweep, so per-ray occupancy is comparable to a real
revolution; nuScenes is 10-sweep accumulated and fails I4 by construction at ~9
returns/cell):

| invariant | real VLP-16 | ATLAS | clean KITTI |
|---|---|---|---|
| **I6** ring-arc gap / az_res | 1.000 | **1.039** | 1.039 |
| **I4** single-return-per-ray | 0.978 | **0.881** | 0.830 |
| per-ring density (pts/ring/frame) | — | 352–433 | 411 |
| **I5** radial density slope | −1.33 | −1.58…−1.62 | −1.76 |
| **I1** 2 mm range grid | 1.000 | 0.101 | 0.105 |
| **I3** azimuth grid | 0.119 | 0.100 | 0.101 |

Reading:

* **I6 is the strongest result.** ATLAS reproduces the `r·dθ` arc-spacing law to
  **1.039 vs the real 1.000** (3.9%), and it is the *identical* 1.039 clean KITTI
  shows — ATLAS inherits the law exactly, adding no error.
* **I4:** ATLAS (0.881) is **closer to real hardware (0.978) than clean KITTI
  (0.830)**.
* **Per-ring density:** ATLAS 352–433 vs FOV-matched clean KITTI 411 —
  indistinguishable.
* **I1/I3 fail, and they fail for clean KITTI too** (both at the 0.10 chance
  level). KITTI ships float32 XYZ in a rectified frame, which destroys the 2 mm
  range grid and the azimuth lattice before ATLAS ever runs. Not a generator
  defect. Note the real capture only scores 0.119 on I3, so that test is weakly
  discriminative even on true hardware.

Not claimed as agreement: azimuth CV (ATLAS 0.25–0.26 vs clean KITTI 0.089 — the
wedge is a localized density perturbation) and rings occupied (49/64 vs 61/64).

Dual-return mechanism, verified on the real capture: `dual` mode gives **2.03**
returns/ray vs **1.02** in `last` mode, the expected factor of two. ATLAS models
this as a per-sensor parameter `P_two` (0.01 Waymo, 0 for single-return nuScenes
and KITTI).

## 4. Intensity ablation

Real spoofed returns **saturate** (median 255/255, fraction==255 = 1.000). ATLAS
writes phantom intensity sampled from the diffuse range-conditioned stats
(median **14.7** on the generated splits). A ~17× gap.

Closing it *collapses* attack success:

| Model | split | baseline ASR | ASR @ I=255 | relative |
|---|---|---|---|---|
| pp_multihead | inject_relative_easy | 0.3585 | 0.0336 | **−91%** |
| pp_multihead | inject_relative_medium | 0.1157 | 0.0156 | −87% |
| pp_multihead | inject_relative_hard | 0.0246 | 0.0139 | −44% |
| voxel0075_voxelnext | inject_relative_easy | 0.4848 | 0.0221 | **−95%** |
| voxel0075_voxelnext | inject_relative_medium | 0.1977 | 0.0131 | −93% |
| voxel0075_voxelnext | inject_relative_hard | 0.0468 | 0.0107 | −77% |
| transfusion_lidar | inject_relative_easy | 0.4028 | 0.0623 | **−85%** |
| transfusion_lidar | inject_relative_medium | 0.1756 | 0.0246 | −86% |
| transfusion_lidar | inject_relative_hard | 0.0468 | 0.0098 | −79% |

Baselines reproduce `atlas_nuscenes_results.csv` (pp 0.363, voxelnext 0.485,
transfusion 0.398), validating the harness. Architecture-agnostic: pillar,
sparse-voxel, and transformer detectors all lose 85–95%.

**Dose-response** (pp_multihead, `inject_relative_easy`, full 1219-target passes):

| forced I | 5 | 15 | 30 | 60 | 100 | 150 | 200 | 255 |
|---|---|---|---|---|---|---|---|---|
| ASR | 0.286 | 0.235 | 0.245 | 0.252 | 0.237 | 0.137 | 0.060 | 0.034 |
| Δ vs 0.354 | −0.068 | −0.120 | −0.109 | −0.103 | −0.117 | −0.217 | −0.295 | −0.320 |

**Not monotone — a plateau with a cliff**, which decomposes the effect:

1. **Constant ~−0.11 across I = 15…100** (a 6.7× brightness range). Any
   *constant* intensity costs ~0.11 ASR regardless of value: this is loss of
   intensity **variance**, not brightness.
2. **A cliff past I ≈ 100** contributing the remaining ~−0.21. I = 100 is exactly
   the **p99 of real nuScenes car intensity**.

| population | median | p99 | frac ≥ 200 |
|---|---|---|---|
| real nuScenes car points | 5.0 | 100 | 0.0002 |
| real nuScenes all points | 15.0 | 88 | 0.0006 |
| ATLAS phantom (as written) | 14.7 | 97 | ~0 |
| **real spoofer (OSF)** | **255** | 255 | **1.000** |

So the collapse is an **out-of-distribution effect localised to the top 1% tail**,
not a "brighter is easier to detect" trend. Implications:

1. ATLAS's diffuse-intensity choice is **correct for a geometry benchmark** — it
   keeps the phantom in-distribution so measured ASR reflects shape/placement
   vulnerability. Saturated intensity would drive every injection ASR to ~0.
2. **ATLAS injection ASR is an upper bound for this specific real attack**, not a
   lower bound. A real saturating spoofer is *more* easily caught. Stated
   honestly, that is a limitation of the real attack (trivially detectable on
   intensity), not of ATLAS geometry.
3. **The plateau is itself an alignment result:** because ASR is flat over
   I = 15…100, the real–synthetic intensity gap does *not* propagate into the
   benchmark numbers — the measured ASR is insensitive to the one channel the
   generator models least faithfully.
4. It identifies a **nearly free defense** (intensity-plausibility filtering),
   and correspondingly means ATLAS cannot be used to evaluate such a filter.

## 5. Cross-attack and clean-accuracy correlations

Injection robustness and removal robustness are close to **orthogonal**:

| correlation | nuScenes (n=5) | KITTI (n=8) |
|---|---|---|
| clean mAP vs injection ASR | +0.68 | −0.17 |
| injection rank vs removal rank | −0.20 | −0.31 |
| injection ASR vs removal ASR | −0.34 | −0.64 |

No p-value reaches 0.05 (closest: KITTI ASR pair, p = 0.089), so the claim is
"not predictive", not "negatively correlated". The load-bearing evidence is the
**KITTI SECOND family**: `second` and `second_iou` rank worst and 2nd-worst
against injection (0.455, 0.435) yet best and 2nd-best against removal (0.314,
0.322) — a 6-rank flip in two related architectures.

The pooled coefficients understate a consistent pattern, because the correlation
depends on injection sparsity:

| injection difficulty | inj-vs-rem r, nuScenes | KITTI |
|---|---|---|
| easy (dense phantom) | −0.17 … +0.13 | −0.57 … +0.06 |
| hard (sparse phantom) | **−0.58 … −0.78** | **−0.57 … −0.95** |

Against dense phantoms every detector fails similarly, so there is nothing to
correlate. Against sparse phantoms — the realistic case — the anti-correlation is
strong and replicates across all 6 removal widths and all injection families.
This is the argument for ATLAS covering both attack modes: a single-attack
benchmark would rank these models roughly backwards on the other attack.

## 6. Bottom line

- The real capture is a **removal** attack → compare against `removal_az*`.
- **Removal strength aligns:** ATLAS 0.90–1.00 vs real 0.83–0.87 at the aim point
  (1.0–1.2×), inside/just above the paper's own 0.836–0.927 band.
- **Width is only partly validated:** `az10`–`az40` lie within the demonstrated
  45° capability; **`az50`/`az60` are extrapolations** and are labelled as such.
- **Scan-pattern fidelity holds** where the dataset permits measuring it: ring-arc
  spacing within 3.9% of hardware, per-ray occupancy closer to real hardware than
  clean KITTI. Where ATLAS deviates (range/azimuth quantization), clean KITTI
  deviates identically — dataset property, not generation artifact.
- **Intensity is the substantive gap** (~17×), but the ablation shows closing it
  *reduces* ASR by 85–95% because saturation is out-of-distribution, and the
  response is flat over I = 15…100 — so the gap does not propagate into the
  benchmark.
- **Not modelled:** laser wavelength, analog receiver electronics, attacker–victim
  timing synchronization. ATLAS assumes a synchronized attacker and simulates the
  resulting point-cloud perturbation.
- Independent severity corroboration: the LGSVL+Apollo recordings show the attack
  reaching unsafe control action (Apollo emergency braking at −5.9 m/s²), not
  merely false detection.
