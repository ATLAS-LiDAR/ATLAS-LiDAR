# ATLAS-Sandbox v1 — a visualization of the ATLAS attack

**v1 scope.** This is an interactive *visualization* of the ATLAS LiDAR spoofing
attack, running on real recorded driving frames. It is a finished artifact, not a
staging ground: attack-success evaluation (a downstream detector), temporal
multi-frame attacks and 3D/multi-ring views are deliberately out of scope and
belong to a later version.

The ATLAS attack in `Polar-Hierarchical-Mamba` is an offline data-generation
pipeline — it perturbs recorded frames and hands the result to a detector. What
the perturbation actually *is* never surfaces: which returns die, which cells a
phantom can occupy, why a given point survives. This shows that, return by
return, on the sensor's own scan lattice.

## What is ATLAS here, and what is not

Being precise about this matters for what the figure can be claimed to show.

**Visualized from ATLAS** — the perturbation model, ported and cross-checked:

| here | from |
|---|---|
| occlusion-aware injection: per-cell range comparison, first-return suppression, dual-return `pTwo` | `nuscenes_atlas_common.spoof_frame` |
| ground anchoring to the 5th percentile of local scene z within 2 m | `ground_anchor_spoof` |
| range-conditioned lognormal intensity, clipped to measured percentiles | `get_stats_for_range` + the `.npy` band statistics |
| beam inclinations, azimuth resolution, intensity bands | the same `atlas_generation/*.npy` assets, emitted by `tools/gen_sensor_assets.py` with source SHAs |
| cell fill and firing-budget metrics | `osf_k6nf2/realizability.py` |

**Added here, because ATLAS does not model it.**
`ATLAS_REAL_WORLD_VALIDATION.md` closes with exactly this list: *"laser
wavelength, analog receiver electronics, attacker-victim timing synchronization.
ATLAS assumes a synchronized attacker and simulates the resulting point-cloud
perturbation."*

- **Time of flight and the relay floor.** A reactive spoofer *d* metres away
  cannot place a phantom closer than *d*; getting inside that needs pulse
  prediction, which is an explicit capability toggle instead of an assumption.
- **Synchronization as a parameter**, at gate-width / firing-period — about 2% for
  a free-running spoofer against a VLP-16.
- **A derived reachable envelope** from power over threshold, divergence and
  standoff, where ATLAS uses fixed `az10`–`az60` widths.
- **Line of sight** from spoofer to victim through the recorded geometry.
- **Removal as a mechanism** — a saturated sub-MOT echo the firmware discards —
  where ATLAS applies a `p_remove` probability.
- **The PRF firing budget**, capping points per revolution.

The short version: **what the attack does to the cloud is ATLAS; whether the
attack is physically possible is new.** That second layer is what lets the figure
label an attack impossible, demonstrated, or an extrapolation, and say why.

## Running it

No build step and no Node required — native ES modules over any static server:

```sh
python3 -m http.server 8080
# open http://127.0.0.1:8080/index.html
```

## Tests

There is no JS runtime on this machine, so the suite runs in headless Chrome and
reports through the DOM. 187 tests, exit code 0 on green:

```sh
./run-tests.sh
```

Verified in both directions — a deliberately failing test reports red and exits
nonzero (see the tripwire procedure in `test/harness.selftest.js`).

## Architecture

```
src/core/     physics, no DOM access
  polar.js          spherical decomposition, (azimuth x beam) cell quantization
  sensor.js         sensor specs; point rate is the primitive, azimuth res derived
  sensor-assets.js  GENERATED - beam inclinations measured from the real datasets
  frame.js          PointFrame: CSR cell-occupancy index over a recorded cloud
  tof.js            time of flight, relay floor, pulse prediction, sync quality
  spoofer.js        spoofer hardware -> reachable cell set, line of sight
  phantom.js        phantom generation by ray-casting reachable directions
  inject.js         occlusion-aware injection (port of ATLAS spoof_frame)
  remove.js         removal via saturated sub-MOT echo (the real PRA mechanism)
  intensity.js      range-conditioned / saturated / constant intensity models
  realizability.js  cell fill, firing budget, demonstrated-envelope verdict
src/io/       .atlasframe reader, raw KITTI/nuScenes upload parsing
src/ui/       render.js (BEV + range image), app.js (state, wiring)
tools/        gen_sensor_assets.py, export_frames.py, parse_dom.py
assets/frames real exported frames + manifest.json
```

Numbers are generated, not transcribed: `tools/gen_sensor_assets.py` emits beam
inclinations and intensity statistics from the same `.npy` assets
`atlas_generation/*_atlas_common.py` loads, with source SHAs in the file header.

## How the added physics works

Detail on the second layer above — the constraints a point-cloud-level model
cannot express.

**Time of flight is real.** A reactive relay spoofer standing *d* metres away
cannot place a phantom closer than *d* — it has to wait for the victim's pulse to
arrive (*d/c*) and its reply has to travel back (*d/c*), so range = *d* +
*c*·(latency+delay)/2. Getting inside that floor requires firing *before* the
victim's pulse, i.e. pulse prediction, which is an explicit capability toggle
rather than an assumption. Timing precision is range precision at 0.15 m/ns.

**Synchronization is a parameter.** An unsynchronized spoofer only lands a pulse
when the victim's receiver gate happens to be open — about 2% of the time on a
VLP-16 (gate width / firing period). That is why every practical attack in the
literature locks to the victim's PRF, and v1 makes that visible.

**The envelope is derived, not dialled.** The MVP had a "projection window"
slider. Here the reachable set follows from power over detection threshold,
divergence and standoff, through a Gaussian receiver-leakage response:
`halfWidth = sigma * sqrt(2 ln R)`, widening as `sqrt(log power)` and shrinking
with distance as `1/d^2`. Line of sight is checked against the recorded geometry,
so a spoofer parked behind a building cannot attack at any power.

**Removal has a mechanism.** Instead of deleting points in a wedge, the attacker
plants a saturated echo inside the sensor's Minimum Operational Threshold (40 cm
on a VLP-16); the sensor prefers the strongest echo and firmware discards
sub-MOT returns, so the genuine return dies with it. Above the MOT the return is
range-*replaced* rather than destroyed — which is exactly what the pre-filter
pcaps show (a 2.01 m target reappearing at ~11 m). A consequence falls out rather
than being assumed: **removal requires pulse prediction**, since a sub-MOT echo
must arrive within ~2.7 ns of the victim firing.

**Every rejected point is accounted for.** Candidates land in exactly one bucket
— `unreachable`, `response`, `tof`, `mot`, `sync`, `cellConflict`, `occluded` —
or are injected, and the buckets reconcile against the request count. One
spoofed return per cell is enforced, so the readout can never disagree with the
picture (an MVP bug) and output is realizable by construction.

## Calibration honesty

The receiver-leakage width `sigma_az` and the reference `powerRatio` are
**jointly fitted** to two measured anchors from the real VLP-16 capture: the
6.8° saturated envelope and the ~0.30 edge/peak per-cell rate ratio (0.25 at the
fringe vs 0.83–0.87 at the aim point). Only the product `sigma*sqrt(2 ln R)` is
identified by the width, so neither constant is an independently measured
quantity. What *is* physically meaningful is the shape (Gaussian falloff) and the
scaling (with power and distance). Vertical leakage is set wide because the real
capture loses all 16/16 channels.

## Findings from building this

Two things surfaced that are worth carrying back to the pipeline.

**1. `osf_k6nf2/realizability.py` uses the wrong azimuth resolution, inflating
cell fill by 2.09×.** It assumes 0.40° for a 10 Hz VLP-16, but the datasheet
gives 0.192° at 10 Hz (300k points/sec over 16 channels; 0.40° corresponds to 20
Hz). Recomputed on `real_last.npz` + `real_dual.npz` (n=126 clusters):

| azimuth res | cell fill p95 (unique) | fill p95 (with dual) | fill max (dual) |
|---|---|---|---|
| 0.40° (as published) | 0.590 | 2.134 | **2.219** |
| 0.192° (datasheet) | 0.283 | 1.025 | 1.065 |

The decisive evidence is the last column: at 0.40° the measured dual-return fill
exceeds **2.219 returns per cell**, above the hard physical limit of 2 for a
dual-return sensor. At 0.192° it is 1.065. Because the published fill envelope is
~2× too permissive, synthetic phantoms compared against it look *more* realizable
than they are. Both sets of figures are kept in `realizability.js` so the
comparison stays explicit.

**2. Derived point rates independently corroborate the measured beam assets.**
Deriving `nBeams * azCellCount * hz` from the fitted azimuth resolutions lands on
the manufacturers' quoted single-return rates: HDL-32E 347,520 vs 350,000 quoted,
HDL-64E 1,333,760 vs 1.33M. That is a strong check that the `.npy` assets are
right.

## Known limitations

Genuine fidelity limits:

- **The scan lattice cannot be recovered exactly from released clouds.** Measured
  on `nuscenes-boston`: 1.63 returns per occupied cell with nearest-inclination
  binning, 1.53 using the dataset's own ring index, against an ideal 1.0. The
  residual is azimuth, not beam assignment — released clouds ship rectified
  float32 XYZ, so the original azimuth grid is gone before anything downstream
  runs (the same effect the validation doc reports as clean KITTI failing the
  azimuth-grid invariant at chance level). Consequence: occupied-cell counts are
  underestimated and cell fill slightly overestimated.
- **Intensity is a substantive gap in ATLAS and remains one here**, though v1 at
  least lets you switch between the in-distribution and saturated models to see
  the trade-off the validation doc quantifies (closing the gap collapses ASR by
  85–95%, because saturation is out-of-distribution in the top 1% tail rather
  than simply brighter).
- **The angular-leakage constants are fitted, not measured** — see *Calibration
  honesty* above.

Out of scope by decision (see **v1 scope** at the top), not oversights:

- **Single frames, no temporal dimension.** The schedule machinery in
  `nuscenes_atlas_common.py` — continuous vs intermittent attack spans — has no
  analogue here.
- **No downstream detector.** This shows what the attack does to the cloud, not
  whether a detector is fooled. Attack *success* remains a question for the
  offline pipeline.
- **No 3D or multi-ring view.** The lattice is modelled in full (azimuth × beam),
  but displayed as a BEV plus a range image rather than a 3D scene.

## Data

`tools/export_frames.py` reads KITTI `velodyne` bins and nuScenes `LIDAR_TOP`
sweeps with their lidar-frame annotations and writes `.atlasframe` (int16
positions at 5 mm, uint8 intensity — ~820 kB for a 120k-point HDL-64E frame).

**Single sweeps only.** nuScenes infos accumulate 10 sweeps, which puts ~9
returns in every cell and destroys the one-return-per-ray property the whole
cell-occupancy model rests on.

```sh
python3 tools/export_frames.py --list kitti      # browse candidates
python3 tools/export_frames.py --library         # export the built-in set
```

Users can also drop in raw frames with no conversion: KITTI `.bin` (stride 4) and
nuScenes `.pcd.bin` (stride 5) are parsed directly in the browser, with the
stride inferred and flagged when genuinely ambiguous.

## Sources

- Cao et al., *You Can't See Me: Physical Removal Attacks on LiDAR-based
  Autonomous Vehicles Driving Frameworks*, USENIX Security 2023
  (arXiv:2210.09482) — the real capture all calibration anchors come from.
- `Polar-Hierarchical-Mamba/ATLAS_REAL_WORLD_VALIDATION.md`,
  `atlas_generation/nuscenes_atlas_common.py`, `osf_k6nf2/realizability.py`.
