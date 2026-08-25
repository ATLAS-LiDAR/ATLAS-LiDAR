/**
 * ATLAS removal, ported faithfully: A-HFR from a stationary roadside spoofer.
 *
 * This is a direct port of atlas_generation/removal.py
 * (`spoof_frame_ahfr_roadside` and friends), because the sandbox's job is to let
 * people play with the *ATLAS* attack, not with an attack of our own design.
 *
 * The algorithm, in order:
 *
 *   1. aim at the target box's centre azimuth/elevation
 *   2. rate-limit the aim change from the previous frame (servo model,
 *      max_az_step_deg = 15) and, by default, hold elevation fixed
 *   3. add Gaussian aim jitter (std 0.05 DEGREES)
 *   4. refuse the frame if the target is inside min_target_range_m = 6
 *   5. draw a frame-level scan gate whose probability is a measured, distance-
 *      banded function of target range (ahfrGateProb - non-monotonic)
 *   6. mask points inside a HARD RECTANGULAR az x el sector (defaults 20 x 16
 *      degrees; az_width is what names the removal_azNN splits)
 *   7. delete each point in the sector independently with probability
 *      p_remove = 0.96
 *
 * HOW THIS DIFFERS FROM remove.js
 * remove.js models the *physical mechanism* from Cao et al. - a saturated
 * sub-MOT echo, with an angular response derived from spoofer power and an
 * azimuth-peaked per-cell rate. ATLAS instead applies a flat p_remove inside a
 * fixed rectangular sector, with no power model, no line of sight and no lattice.
 * Both are in the sandbox on purpose: this module is what ATLAS actually does,
 * remove.js is what the hardware actually does. They are not interchangeable and
 * should not be averaged together.
 */
import { cartToSpherical } from "./polar.js";

export const AHFR_DEFAULTS = Object.freeze({
  azWidthDeg: 20.0,
  elWidthDeg: 16.0,
  pRemove: 0.96,
  maxAzStepDeg: 15.0,
  fixElevation: true,
  centerJitterStdDeg: 0.05,
  minTargetRangeM: 6.0,
  allowCloseRangeExtrapolation: false,
  // select_initial_target
  desiredRange: 30.0,
  tol: 5.0,
  targetClass: 1,
  minRange: 20.0,
});

const DEG = Math.PI / 180;

/** numpy-ish rng with uniform and normal draws, seeded and reproducible. */
export function makeAtlasRng(seed = 42) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    random: uniform,
    normal(mu = 0, sigma = 1) {
      let u = uniform();
      if (u <= 0) u = Number.MIN_VALUE;
      return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform());
    },
  };
}

/** Map any angle into [-pi, pi]. */
export function wrapToPi(a) {
  return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

/**
 * Frame-level activation probability by target distance.
 *
 * Verbatim from removal.py, including two properties worth preserving: the
 * bands are NOT monotonic (0.66 at 10-20 m, 0.46 at 20-30 m) because they are
 * averages of measured ranges, and the `<= upper and >= lower` chain resolves an
 * exact boundary to the LOWER band.
 */
export function ahfrGateProb(distance) {
  if (distance < 10) return 0.38; // avg of 27-44%
  if (distance <= 20 && distance >= 10) return 0.66; // avg of 61-72%
  if (distance <= 30 && distance >= 20) return 0.46; // avg of 40-50%
  if (distance <= 40 && distance >= 30) return 0.82; // avg of 73-87%
  if (distance <= 50 && distance >= 40) return 0.93; // avg of 81-100%
  return 1.0;
}

/** Azimuth (wrapped to [-pi, pi]) and elevation of a box centre. */
export function boxCenterToAzEl(box) {
  const s = cartToSpherical(box[0], box[1], box[2]);
  return { az: wrapToPi(s.az), el: s.el };
}

export function boxCenterRange(box) {
  return Math.hypot(box[0], box[1], box[2]);
}

/** Step from prev toward next, limited to maxStepRad, taking the short way. */
export function clampAngleStep(prevAngle, newAngle, maxStepRad) {
  let delta = wrapToPi(newAngle - prevAngle);
  if (delta > maxStepRad) delta = maxStepRad;
  else if (delta < -maxStepRad) delta = -maxStepRad;
  return wrapToPi(prevAngle + delta);
}

/**
 * Points inside the A-HFR sector: a hard rectangle in (azimuth, elevation).
 * @returns {Uint8Array} 1 = inside
 */
export function sectorMask(frame, azCenter, elCenter, azWidthDeg, elWidthDeg) {
  const azHalf = (azWidthDeg * DEG) / 2;
  const elHalf = (elWidthDeg * DEG) / 2;
  const m = new Uint8Array(frame.count);
  for (let i = 0; i < frame.count; i++) {
    const dAz = wrapToPi(frame.az[i] - azCenter);
    const dEl = frame.el[i] - elCenter;
    m[i] = Math.abs(dAz) <= azHalf && Math.abs(dEl) <= elHalf ? 1 : 0;
  }
  return m;
}

/**
 * Pick the vehicle to erase.
 *
 * Boxes are [x, y, z, dx, dy, dz, yaw, ..., class] with the class in the last
 * column (1 = vehicle, 2 = pedestrian, 3 = cyclist).
 *
 * NOTE ON FIDELITY: within the tolerance band ATLAS selects with
 * `argmax(|dist - desired_range|)`, i.e. the candidate FURTHEST from the desired
 * range. That reads like it was meant to be argmin, but changing it would change
 * which vehicle every removal split targets, so the port keeps ATLAS's
 * behaviour. Pass `preferNearestToDesired: true` to opt into the other reading.
 */
export function selectInitialTarget(gtBoxes, {
  desiredRange = AHFR_DEFAULTS.desiredRange,
  tol = AHFR_DEFAULTS.tol,
  targetClass = AHFR_DEFAULTS.targetClass,
  minRange = AHFR_DEFAULTS.minRange,
  classCol = -1,
  negative = false,
  preferNearestToDesired = false,
} = {}) {
  if (!gtBoxes || gtBoxes.length === 0) return { index: null, box: null };

  const dist = [];
  const cand = [];
  for (let i = 0; i < gtBoxes.length; i++) {
    const b = gtBoxes[i];
    const d = Math.hypot(b[0], b[1], b[2]);
    dist.push(d);
    const label = Math.round(classCol < 0 ? b[b.length + classCol] : b[classCol]);
    const inFront = negative ? true : b[0] > 0;
    if (inFront && d > minRange && label === targetClass) cand.push(i);
  }
  if (cand.length === 0) return { index: null, box: null };

  const band = cand.filter((i) => Math.abs(dist[i] - desiredRange) <= tol);
  let idx;
  if (band.length === 0) {
    // fallback: the furthest qualifying object
    idx = cand.reduce((best, i) => (dist[i] > dist[best] ? i : best), cand[0]);
  } else if (preferNearestToDesired) {
    idx = band.reduce(
      (best, i) => (Math.abs(dist[i] - desiredRange) < Math.abs(dist[best] - desiredRange) ? i : best),
      band[0]
    );
  } else {
    idx = band.reduce(
      (best, i) => (Math.abs(dist[i] - desiredRange) > Math.abs(dist[best] - desiredRange) ? i : best),
      band[0]
    );
  }
  return { index: idx, box: gtBoxes[idx].slice() };
}

/** Volume of a box [x,y,z,dx,dy,dz,...]. */
function boxVolume(b) {
  return Math.abs(b[3] * b[4] * b[5]);
}

/**
 * Track the target across frames: nearest-neighbour matching against a
 * constant-velocity prediction, with spatial, class and size gates. Port of
 * ATLAS's match_with_velocity.
 *
 * Boxes must be [x, y, z, dx, dy, dz, yaw, vx, vy, ..., class] - velocity in
 * columns 7 and 8 and the class in the LAST column, which is how ATLAS indexes
 * them.
 *
 * `dt` MATTERS. ATLAS defaults to 0.1 for Waymo, but nuScenes KEYFRAMES are
 * 2 Hz, so callers there must pass 0.5. Its own docstring warns that otherwise
 * the prediction is 5x too short and the residual for a fast vehicle exceeds
 * maxStep, silently dropping the track.
 *
 * @returns {{box: number[]|null, index: number|null, dist: number|null}}
 */
export function matchWithVelocity(prevBox, prevPrevBox, currBoxes, {
  targetClass = 1,
  maxStep = 6.0,
  maxVolumeRatio = 2.0,
  classCol = -1,
  dt = 0.1,
} = {}) {
  const none = { box: null, index: null, dist: null };
  if (!currBoxes || currBoxes.length === 0 || !prevBox) return none;

  const cls = (b) => Math.round(classCol < 0 ? b[b.length + classCol] : b[classCol]);
  const valid = [];
  for (let i = 0; i < currBoxes.length; i++) {
    if (cls(currBoxes[i]) === targetClass) valid.push(i);
  }
  if (valid.length === 0) return none;

  // constant-velocity prediction from the previous box's own velocity
  const vx = prevBox[7] ?? 0;
  const vy = prevBox[8] ?? 0;
  const pred = [prevBox[0] + vx * dt, prevBox[1] + vy * dt, prevBox[2]];

  const scored = valid
    .map((i) => ({
      i,
      d: Math.hypot(currBoxes[i][0] - pred[0], currBoxes[i][1] - pred[1], currBoxes[i][2] - pred[2]),
    }))
    .sort((a, b) => a.d - b.d);

  const prevVol = boxVolume(prevBox);
  for (const { i, d } of scored) {
    if (d > maxStep) break; // sorted, so everything after is farther
    if (prevVol > 0) {
      const v = boxVolume(currBoxes[i]);
      const ratio = Math.max(v, prevVol) / Math.min(v, prevVol);
      if (ratio > maxVolumeRatio) continue;
    }
    return { box: currBoxes[i].slice(), index: i, dist: d };
  }
  return none;
}

/**
 * One frame of A-HFR removal.
 *
 * @param {object} frame        PointFrame (uses its precomputed az/el)
 * @param {number[]} targetBox  the real actor being erased, used only to aim
 * @param {object|null} state   {azCenter, elCenter} carried between frames
 * @param {boolean} forceActive skip the scan-gate draw (for tests/визual clarity)
 * @returns {{keepMask: Uint8Array, state: object, meta: object}}
 */
export function ahfrRemoval({
  frame,
  targetBox,
  state = null,
  azWidthDeg = AHFR_DEFAULTS.azWidthDeg,
  elWidthDeg = AHFR_DEFAULTS.elWidthDeg,
  pRemove = AHFR_DEFAULTS.pRemove,
  maxAzStepDeg = AHFR_DEFAULTS.maxAzStepDeg,
  fixElevation = AHFR_DEFAULTS.fixElevation,
  centerJitterStdDeg = AHFR_DEFAULTS.centerJitterStdDeg,
  minTargetRangeM = AHFR_DEFAULTS.minTargetRangeM,
  allowCloseRangeExtrapolation = AHFR_DEFAULTS.allowCloseRangeExtrapolation,
  forceActive = false,
  rng = null,
}) {
  const r = rng || makeAtlasRng(42);
  const st = state || {};

  const { az: rawAz, el: rawEl } = boxCenterToAzEl(targetBox);
  const targetRangeM = boxCenterRange(targetBox);
  const scanGateProb = ahfrGateProb(targetRangeM);

  let azCenter =
    st.azCenter === undefined
      ? rawAz
      : clampAngleStep(st.azCenter, rawAz, maxAzStepDeg * DEG);
  const elCenter =
    st.elCenter === undefined ? rawEl : fixElevation ? st.elCenter : rawEl;

  // jitter is drawn in DEGREES then converted, as in removal.py
  azCenter = wrapToPi(azCenter + r.normal(0.0, centerJitterStdDeg) * DEG);

  const newState = { azCenter, elCenter };
  const keepMask = new Uint8Array(frame.count).fill(1);

  const baseMeta = {
    azCenterRad: azCenter,
    elCenterRad: elCenter,
    azWidthDeg,
    elWidthDeg,
    pRemove,
    scanGateProb,
    targetRangeM,
    rawTargetAzRad: rawAz,
    rawTargetElRad: rawEl,
    maxAzStepDeg,
    fixElevation: !!fixElevation,
    mode: "roadside",
    nPointsBefore: frame.count,
    nPointsInSector: 0,
    nPointsRemoved: 0,
    nPointsAfter: frame.count,
    activeThisFrame: false,
    rangeGatePassed: true,
    extrapolatedCloseRange: false,
    inactiveReason: null,
  };

  if (targetRangeM < minTargetRangeM && !allowCloseRangeExtrapolation) {
    return {
      keepMask,
      state: newState,
      meta: { ...baseMeta, rangeGatePassed: false, inactiveReason: "target_too_close" },
    };
  }

  const active = forceActive || r.random() < scanGateProb;
  if (!active) {
    return {
      keepMask,
      state: newState,
      meta: { ...baseMeta, inactiveReason: "scan_gate" },
    };
  }

  const mask = sectorMask(frame, azCenter, elCenter, azWidthDeg, elWidthDeg);
  let nInSector = 0;
  for (let i = 0; i < frame.count; i++) if (mask[i]) nInSector++;

  const meta = {
    ...baseMeta,
    activeThisFrame: true,
    nPointsInSector: nInSector,
    extrapolatedCloseRange:
      targetRangeM < minTargetRangeM && allowCloseRangeExtrapolation,
  };

  if (nInSector === 0) {
    return { keepMask, state: newState, meta: { ...meta, inactiveReason: "empty_sector" } };
  }

  let removed = 0;
  for (let i = 0; i < frame.count; i++) {
    if (!mask[i]) continue;
    if (r.random() < pRemove) {
      keepMask[i] = 0;
      removed++;
    }
  }

  return {
    keepMask,
    state: newState,
    meta: {
      ...meta,
      nPointsRemoved: removed,
      nPointsAfter: frame.count - removed,
      sectorMask: mask,
    },
  };
}
