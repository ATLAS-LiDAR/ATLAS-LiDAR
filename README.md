# ATLAS

A LiDAR spoofing attack, and an interactive sandbox for it.

Two halves, in one repository on purpose:

| | | |
|---|---|---|
| **`atlas/`** | the attack | The reference implementation: A-HFR sector removal and trace-based injection. ~1,400 lines, **numpy only** - no torch, no CUDA, no OpenPCDet, no checkpoints. `pip install -e .` |
| **everything else** | the sandbox | An interactive visualizer for it, running on real recorded driving frames. No build step, no Node. |

They live together because that is what makes the fidelity claim checkable. The
sandbox is a JavaScript port of `atlas/`, and `tools/crossvalidate.py` runs both on
identical input and compares integer outcomes: **129 of 129 configurations agree
exactly**. With the two halves in separate repos that check needed a sibling clone
and a specific conda env, so it only ever ran on one machine. Here it runs in CI.

`atlas/` was copied byte for byte out of the paper's pipeline repository, and
`atlas/PROVENANCE.json` pins a hash for every file, so agreeing with it is still
transitively agreeing with the published implementation rather than with a local
edit. See [atlas/PROVENANCE.md](atlas/PROVENANCE.md).

## What is not here

Generating the full attacked datasets and scoring them against real detectors needs
OpenPCDet, torch, trained checkpoints and the raw KITTI/nuScenes archives. That half
stays in the paper's pipeline repo. The sandbox writes `.atlasattack` bundles and
`tools/import_attack.py` converts them into what the detectors eat, so the round
trip works - it just needs that setup locally.

## Sandbox versions

| | | |
|---|---|---|
| **v1** | tag `v1` | Barebones demo. The scan-lattice model, spoofer hardware physics and realizability verdict, on recorded frames. Frozen and citable. |
| **v2** | `main` | **The ATLAS visualizer.** ATLAS's own algorithms, ported: trace-based injection and A-HFR removal, with the sensor spinning in real time. This is where the project is now. |
| **v3** | planned | The true ATLAS-Sandbox: visualize *any* raw-signal spoofing attack, not just ATLAS. |

v2's spec is **fidelity**. Every attack it shows is a port of ATLAS's own code, so
what you see is what the pipeline generates - not an approximation that looks
similar. Where ATLAS does something surprising, the port keeps the surprise and
documents it rather than quietly tidying it up.

## Running it

**The sandbox** needs no build step and no Node - native ES modules over any
static server:

```sh
python3 -m http.server 8080
# open http://127.0.0.1:8080/index.html
```

**The attack** is a normal Python package, numpy and nothing else:

```sh
pip install -e .
python3 -c "from atlas import removal; print(removal.spoof_frame_ahfr_roadside)"
```

A configuration is shareable, and headless-drivable, through the URL:

```
index.html?mode=remove&scene=kitti-parked-row&azWidth=40&forceActive=1
index.html?mode=inject&difficulty=hard&spin=0
index.html?scene=nuscenes-scene0&placement=global&frame=12
index.html?scene=nuscenes-scene0&schedule=intermittent
index.html?scene=nuscenes-scene0&attacker=dynamic&frame=8
index.html?sweptTo=120            # freeze the sweep mid-fill
```

## What v2 shows

**Injection** samples a *real recorded vehicle* from `trace_{easy,medium,hard}`,
which is what those split names mean - phantom point density, measured at median
186 / 26 / 7 points on the nuScenes libraries. The trace is de-rotated and
centred (`center_trace`), yawed by `uniform(-0.2, 0.2)`, offset `uniform(15,20)` m
ahead and `uniform(-3,3)` m lateral (`place_trace_relative`), anchored to the
local ground, given range-conditioned intensity, and resolved against the real
cloud per (azimuth x beam) cell (`spoof_frame`): a phantom behind a real surface
is dropped, a phantom in front destroys what was behind it.

**Removal** aims a roadside A-HFR spoofer at a vehicle you pick - click it, or let
ATLAS choose with `select_initial_target`. The aim is rate-limited like a servo
(15 deg/frame), elevation is held fixed, and a frame-level scan gate fires with a
probability that is a *measured, non-monotonic* function of target range. Inside a
hard 20 deg x 16 deg sector every return dies with probability `p_remove` = 0.96.
`az_width` is the slider that names the `removal_azNN` splits.

**The sweep** reveals points in the order the sensor actually acquired them. A
recorded cloud has already been sampled, so there is nothing left to raycast -
but the scan lattice was recovered, so the animation walks a prefix of the
azimuth-sorted index, which is an O(1) lookup per rendered frame.

## Driving sequences

Static single frames remain the default. Selecting a **driving sequence** instead
plays ATLAS's attack across keyframes, which is where its temporal machinery
becomes visible.

**Sequences are nuScenes only.** KITTI's object-detection split on disk is a set
of independent frames with no temporal ordering and no ego poses, so it cannot
make one. nuScenes keyframes carry `ref_from_car` and `car_from_global`, which
compose into ego->global. The cost is a sparser 32-beam cloud (~34k points vs
KITTI's ~120k).

Two placement policies, and the difference only exists in a sequence:

| | |
|---|---|
| **Follows ego** (`place_trace_relative`) | the phantom sits 15-20 m ahead in *every* frame, so it travels with the vehicle |
| **World-fixed** (`place_trace_global`) | placed once in the anchor keyframe, lifted into world coordinates, and projected back down into each later frame - so the ego drives past it, and it recedes and gets more obliquely occluded as it goes |

**Removal tracks its target** across keyframes with `match_with_velocity`: a
constant-velocity prediction gated on distance, class and size. The aim is
rate-limited between frames like a servo, so the sector visibly sweeps to follow
a moving vehicle, and the readout says when the track is **LOST**.

**The attack schedule** (`build_single_attack_schedule`) can be every keyframe, a
continuous burst, or intermittent - one attack per sequence, flickering on and off
with the attacked-frame count held equal to the continuous case.

**Sequences are exported at full scene length** (40 and 41 keyframes, ~20 s each),
and that is a correctness requirement rather than a nicety. Intermittent mode needs
`nFrames - cleanPrefix` to exceed the longest attack it can draw, or `L` fills the
whole available span, no gap fits, and the schedule is flagged `degenerate`. ATLAS's
`maxLen = 32` was chosen against full nuScenes scenes: 40 keyframes minus a 4-frame
clean prefix leaves 36 > 32, so it can never degenerate. An earlier half-length
20-keyframe export left only 16 available, and any `requestedL >= 16` collapsed -
over half of all seeds. `test/pose.test.js` asserts the requirement against the
manifest rather than pinning the frame count. The cost is asset size: ~10 MB per
sequence, 20 MB total.

One display simplification, stated because it is not physical: **one displayed
revolution advances one keyframe.** nuScenes keyframes are 2 Hz samples of a 20 Hz
spin, so the real gap between them is ten revolutions. Compressing it keeps the
two clocks legible at once.

## The attacker's hardware

ATLAS models no attacker position at all - injection is a trace placed in the
cloud, removal is a sector aimed at a target - so drawing hardware means choosing
geometry ATLAS does not specify. Two mountings:

| | |
|---|---|
| **Roadside** | a box on the ground. It holds ONE world position, so as you drive past its bearing and range both change: it approaches, is passed, and recedes. That apparent motion is the ego moving, not the attacker. |
| **Follows ego** | vehicle-mounted or a chase vehicle. Fixed offset from the ego, so its geometry never changes. An ego-fixed phantom (`place_trace_relative`, always 15-20 m ahead) implicitly needs this mounting. |

Shift-click the left view to plant it. Its position does **not** change the attack
unless the physics overlay is on; it reports whether real hardware standing there
could have produced what ATLAS generated.

**Two views, toggled: ATLAS output vs producible.** A spoofed return only exists
if the attacker's light reached the detector while that column's beam was
listening, so a column outside its reach can never carry one - whatever ATLAS
generated there. The toggle switches between showing every point ATLAS produced
and dimming the ones a box standing here could not have made. Neither setting
changes what is generated; the attack itself is only gated by the physics overlay.

Driving past a roadside attacker with a world-fixed phantom, this is what happens
to the same ATLAS output:

| keyframe | attacker | reach | producible |
|---|---|---|---|
| 0 | 18.0 m, 11° | ±5.2° | **141 / 141** |
| 6 | 27.0 m, −46° | ±4.4° | 10 / 15 |
| 12 | 47.8 m, −66° | ±2.9° | **0 / 38** |
| 18 | 55.2 m, −69° | ±2.3° | 0 / 5 |

By six seconds in, a stationary roadside box can produce **none** of the phantom
ATLAS is still placing - it has been left behind and its reach no longer covers
it. Azimuths are sequential (each is a separate firing, so the phantom fills in as
the beam crosses the band) while the elevations of one column are simultaneous,
which is why the real capture loses all 16 channels at once.

**When the attacker can actually be heard.** The victim's receiver is
co-boresighted with its emitter, so at any instant it only accepts light arriving
within a narrow cone around the direction the beam currently points - 0.11° on an
HDL-64E. That keyhole is drawn at the beam, and the attacker's reachable band is
drawn around **its own** bearing. The attacker is only heard while the one sweeps
through the other, which for a ±5° band is **2.9% of each revolution**. Outside
that window its light cannot enter the receiver at all, however hard it fires -
so the laser goes dim and the readout says *shut out*.

That window is keyed to the attacker's bearing, not the attacked sector's. The two
coincide only when the attacker happens to sit on the sector; a roadside attacker
drifts off it.

**Alignment is not the criterion, containment is.** An attacker does not have to
sit on the azimuth it injects at: its beam saturates a band around its own
bearing, 6.8° full width at the reference power in Cao et al., widening as
`sqrt(log power)` and quoted up to a demonstrated 45° maximum attack angle. So the
readout asks what fraction of the attacked sector falls inside that band, and
drift degrades coverage gradually instead of flipping a boolean.

Two consequences worth watching, both measured in the sandbox:

- **The reachable band narrows as the attacker recedes**, because received power
  goes as 1/d². Driving past a roadside attacker on a real sequence: standoff
  18 → 54 m, bearing 11° → −69°, reach ±5.2° → ±2.4°, coverage 100% → ~45%. For a
  world-fixed phantom sited near the attacker the two recede *together*, so the
  angular offset stays small and the coverage loss is mostly radial - a power
  problem, not an aiming problem.
- **ATLAS's default 20° removal sector is ~3× the measured 6.8° envelope.** Even
  perfectly aimed at reference power, a real box covers only about a third of it,
  and the wider `removal_azNN` splits less still - consistent with
  ATLAS_REAL_WORLD_VALIDATION.md labelling az50/az60 capability extrapolations.

## Where the constants come from

Every ATLAS constant the sandbox uses is transcribed from the source, and pinned
by `test/atlas-constants.test.js` so drift is caught. The ones that are easy to get
wrong:

| | value | source |
|---|---|---|
| `p_remove` | **derived from `az_width`**: `{10:0.97, 20:1.00, 30:0.99, 40:0.98, 50:0.90, 60:0.90}` | both drivers override `removal.py`'s 0.96 default with this table |
| allowed `az_width` | only 10/20/30/40/50/60 | `argparse choices=sorted(PRR)` |
| injection window, nuScenes | 15–20 m ahead, ±3 m lateral, yaw ±0.2 | `place_trace_relative` |
| injection window, **KITTI** | **12–30 m ahead, ±6 m lateral, yaw ±0.3** | `kitti_atlas_common.place_trace` — wider, set by the camera FOV |
| removal target, **KITTI** | **nearest car in front beyond 10 m, by footprint range** — not `select_initial_target` at all | `generate_data_kitti.py` |
| removal target, nuScenes | `desired_range=40`, `min_range=10`, `negative=True` | `generate_data_removal_nuscenes.py` |
| tracking, nuScenes | `dt=0.5`, `max_step=12.0` | `NUSC_KEYFRAME_DT`, `NUSC_MAX_STEP_M` |
| tracking, Waymo (for contrast) | `dt=0.1`, `max_step=6.0`, `desired_range=60` | `generate_data_removal.py` |
| world-fixed anchor | the **last** frame of the attack span | `anchor_pose = idx_to_pose[max(span)]` |
| phantom intensity | **one** band from the phantom's mean range, applied to every point *before* occlusion; left untouched past the last band | `r_avg = avg_spoof_range(spf)` then `get_stats_for_range` |
| track lost | **gives up** for the rest of the sequence | `if new_box is None: break` |

Not transcribable: **numpy's RNG stream cannot be reproduced in JavaScript**, so a
given seed selects a different trace, offset and yaw than the pipeline would. Draw
order and distributions match, so behaviour is statistically the same and
self-consistent, but never bit-identical. The trace pool is also a 500-trace subset
of each library (the full count is recorded in each file's header).

## Fidelity notes

Things ATLAS does that a tidier implementation would get wrong, kept and
documented in the ports:

- `ahfr_gate_prob` is **not monotonic** in distance (0.66 at 10-20 m, 0.46 at
  20-30 m) because the bands are measurement averages; its `<= upper and >= lower`
  chain resolves an exact boundary to the **lower** band.
- `select_initial_target` picks, within the tolerance band,
  `argmax(|dist - desired_range|)` - the candidate **furthest** from the desired
  range. That reads like an intended `argmin`, but changing it would change which
  vehicle every removal split targets, so it is ported as-is with an opt-in flag
  for the other reading.
- `center_trace` transforms **xy only**; z keeps the height at which the object
  was originally observed and is corrected later by ground anchoring. Recentring z
  would double-correct and sink the phantom.
- nuScenes velocity is a finite difference against the neighbouring keyframe, so an
  object with no match there gets **NaN**. pcdet zeroes it before ATLAS sees a box
  (`gt_boxes[np.isnan(gt_boxes)] = 0` under `SET_NAN_VELOCITY_TO_ZEROS`, which every
  nuScenes config in the pipeline enables), so the exporter does the same. Left
  alone it would poison `match_with_velocity`'s prediction - and, because
  `json.dumps` writes a bare `NaN`, silently produce a header no browser can parse.
  The exporters now pass `allow_nan=False` so that fails loudly instead.
- `match_with_velocity` defaults to `dt=0.1` for Waymo, but nuScenes keyframes are
  2 Hz. Its own docstring warns that using 0.1 there makes the constant-velocity
  prediction 5x too short, so a fast vehicle exceeds `max_step` and the track is
  dropped **silently**. The sequence header carries `dtSec = 0.5` and the port
  uses it.
- The aim jitter is applied AFTER the servo clamp, so an observed frame-to-frame
  aim step can exceed `max_az_step_deg` by a few jitter sigma. The clamp bounds
  the commanded step, not the realized one.
- `spoof_frame` evaluates each phantom point **independently, with no per-cell
  dedupe**, so an ATLAS phantom can carry more than one return per (az, beam)
  cell. The sandbox reports it (a typical easy-trace phantom lands around 2.4
  returns/cell) rather than silently fixing it - that is precisely the question
  `osf_k6nf2/realizability.py` was written to ask.

## Cross-validation against ATLAS's own code

Reading two implementations and asserting they agree is a weak method — it had
already produced eight wrong constants here. `tools/crossvalidate.py` measures
agreement instead: it imports ATLAS's real functions
(`kitti_atlas_common.spoof_frame`, `nuscenes_atlas_common.spoof_frame`,
`removal.spoof_frame_ahfr_roadside`) and runs them and the sandbox on the same
exported frame, the same trace and the same parameters, then compares integer
outcomes — points requested, points injected, real returns destroyed, sector
occupancy, and a checksum of the placed phantom.

Both sides are made deterministic rather than statistically compared: every
random draw is pinned to its extreme (`ZeroRng` / `zeroRng`), the trace index,
offset and yaw are passed explicitly, `p_remove = 1`, the scan gate is forced
open, and aim jitter is zero. On a single-return sensor `P_TWO = 0`, so
occlusion resolution has nothing left to draw either.

```
$ tools/crossvalidate.py
129 agree, 0 disagree, 0 skipped
```

The grid spans 3 frames × 5 traces × 5 placements plus all 6 tabulated wedge
widths × 3 targets. It found two bugs that reading could not:

- **`np.percentile(z, 5)` interpolates.** ATLAS's ground anchor takes the 5th
  percentile of local scene z, and numpy's default method interpolates linearly
  between the two straddling samples. Taking the lower sample shifted `dz` by
  ~3 mm, which pushed six phantom points onto the next beam down and destroyed a
  different set of real returns. `percentile5` now reproduces numpy exactly.
- **float32 storage changed occlusion outcomes.** Positions were stored as
  float32 while ATLAS works in float64, so a range that ties a real return
  exactly in ATLAS came out marginally smaller here, taking the "phantom in
  front" branch and destroying one extra point. The loaders and `PointFrame` now
  keep positions in float64.

Not covered: intensity (drawn at random on both sides, and geometrically inert),
and anything downstream of numpy's RNG stream, which cannot be reproduced in
JavaScript.

## The physics overlay is not ATLAS

v1's model - power-derived reachable envelope, line of sight, time-of-flight relay
floor, synchronization, PRF budget - is available as an **opt-in overlay** and is
**off by default**. ATLAS assumes a synchronized attacker and applies none of it,
so turning it on shows a stricter, more physically honest attack that ATLAS itself
would not generate. Leaving it off is what makes this a visualizer *of ATLAS*.

## Tests

There is no JS runtime on this machine, so the suite runs in headless Chrome and
reports through the DOM. 451 tests, exit code 0 on green:

```sh
./run-tests.sh                        # 451 unit tests
python3 tools/crossvalidate.py        # 129 configs vs ATLAS's own functions
python3 tools/check_provenance.py     # atlas/ is unmodified
bash tools/smoke.sh "" "mode=remove"  # UI loads without console errors
```

Verified in both directions — a deliberately failing test reports red and exits
nonzero (see the tripwire procedure in `test/harness.selftest.js`).

All four run in CI (`.github/workflows/ci.yml`) on numpy plus the runner's Chrome:
no conda env, no GPU, no raw datasets. That is what extracting `atlas/` into this
repo bought. The provenance check runs *before* cross-validation deliberately — a
129/129 against an edited reference implementation would be worthless.

## Architecture

```
src/core/
  ATLAS ports (v2 - what the visualizer shows by default)
    atlas-inject.js   center_trace, place_trace_relative, place_trace_global
    schedule.js       build_single_attack_schedule (continuous / intermittent)
  attacker placement (visualization only; ATLAS specifies no position)
    attacker.js       roadside vs vehicle-mounted, envelope containment
    pose.js           rigid ego<->global transforms for sequences
    atlas-removal.js  A-HFR: spoof_frame_ahfr_roadside, ahfr_gate_prob,
                      select_initial_target, the rectangular sector
    inject.js         spoof_frame occlusion; atlasFaithful mode skips v1's gates
    intensity.js      range-conditioned / saturated / constant intensity
    sweep.js          azimuth-ordered rotation state
  lattice + sensor model (shared)
    polar.js          spherical decomposition, (azimuth x beam) quantization
    sensor.js         sensor specs; point rate primitive, azimuth res derived
    sensor-assets.js  GENERATED - beam inclinations measured from the datasets
    frame.js          PointFrame: CSR cell-occupancy index, azimuth-sorted
  v1 physics (opt-in overlay, NOT ATLAS)
    spoofer.js        hardware -> reachable cell set, line of sight
    tof.js            relay floor, pulse prediction, sync quality
    remove.js         Cao et al. sub-MOT saturation mechanism
    phantom.js        ray-cast phantom shapes
    realizability.js  cell fill, firing budget, demonstrated-envelope verdict
  v3 groundwork (built early; not part of v2's story)
    iou.js            oriented 3D IoU + box hit-testing
    detect.js         PROXY detector - geometric detectability only
    compose.js        the cloud a victim would report after an attack
src/io/       .atlasframe frames, .atlastrace libraries, upload parsing, export
src/ui/       render.js (BEV + range image + sweep), app.js (state, wiring)
src/io/       .atlasframe frames, .atlasseq sequences, .atlastrace libraries
tools/        gen_sensor_assets.py, export_frames.py, export_traces.py,
              export_sequence.py, export_headless.py, import_attack.py,
              run_eval.sh, parse_dom.py
assets/       frames/ (5 static frames) + traces/ (6 trace libraries)
              + sequences/ (2 nuScenes driving sequences)
```

Numbers are generated, not transcribed: `tools/gen_sensor_assets.py` emits beam
inclinations and intensity statistics from the same `.npy` assets
`atlas/*_atlas_common.py` loads, with source SHAs in the file header.

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
- `atlas/nuscenes_atlas_common.py` and `atlas/removal.py` (in this repo; see
  [atlas/PROVENANCE.md](atlas/PROVENANCE.md)). For the evaluation half:
  `Polar-Hierarchical-Mamba/ATLAS_REAL_WORLD_VALIDATION.md` and
  `osf_k6nf2/realizability.py` in the pipeline repo.
