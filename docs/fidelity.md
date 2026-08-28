# Fidelity: how the port is held to ATLAS

The sandbox is a JavaScript port of `atlas/`. This is the evidence that it agrees,
and the places where it deliberately does not.

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

## Calibration honesty

The receiver-leakage width `sigma_az` and the reference `powerRatio` are
**jointly fitted** to two measured anchors from the real VLP-16 capture: the
6.8° saturated envelope and the ~0.30 edge/peak per-cell rate ratio (0.25 at the
fringe vs 0.83–0.87 at the aim point). Only the product `sigma*sqrt(2 ln R)` is
identified by the width, so neither constant is an independently measured
quantity. What *is* physically meaningful is the shape (Gaussian falloff) and the
scaling (with power and distance). Vertical leakage is set wide because the real
capture loses all 16/16 channels.

