# The sandbox

How the visualizer works and what it shows. Start it with `python3 -m http.server 8080`
from the repo root, then open <http://127.0.0.1:8080/index.html>.

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

## The physics overlay is not ATLAS

v1's model - power-derived reachable envelope, line of sight, time-of-flight relay
floor, synchronization, PRF budget - is available as an **opt-in overlay** and is
**off by default**. ATLAS assumes a synchronized attacker and applies none of it,
so turning it on shows a stricter, more physically honest attack that ATLAS itself
would not generate. Leaving it off is what makes this a visualizer *of ATLAS*.

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

