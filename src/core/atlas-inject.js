/**
 * ATLAS injection placement, ported faithfully.
 *
 * ATLAS does not inject a synthetic shape. It samples a REAL recorded object
 * point cloud (a "trace"), de-rotates and centres it, applies a small random
 * yaw, and offsets it in front of the ego. Port of `center_trace` and
 * `place_trace_relative` from atlas_generation/nuscenes_atlas_common.py.
 *
 * The occlusion resolution that follows (`spoof_frame`), ground anchoring and
 * range-conditioned intensity already live in inject.js and intensity.js.
 *
 * `place_trace_global` is deliberately not ported: it lifts the phantom into
 * world coordinates so a fixed location persists across frames, which only means
 * something once the sandbox has multiple frames. Static single frames use the
 * relative placement.
 *
 * ON REPRODUCIBILITY: numpy's Generator stream cannot be reproduced in JS, so a
 * given seed will not select the same trace as the Python pipeline. The draw
 * ORDER and the distributions are identical, so the behaviour is statistically
 * the same and self-consistent within the sandbox.
 */

/** ATLAS's ego-relative placement window. */
export const RELATIVE_OFFSET = Object.freeze({
  xMin: 15,
  xMax: 20,
  yAbs: 3,
  yawAbs: 0.2,
});

/** Seeded rng with the draws the placement needs. */
export function makeAtlasRng(seed = 42) {
  let a = (seed >>> 0) || 1;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    random: next,
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    integers: (n) => Math.min(n - 1, Math.floor(next() * n)),
  };
}

/**
 * Move a trace to the origin, ego-aligned (yaw 0).
 *
 * Only xy is transformed: z keeps the height at which the object was originally
 * observed, and is corrected later by ground anchoring. Recentring z here would
 * double-correct and sink the phantom into the road.
 */
export function centerTrace(trace) {
  const [bx, by] = trace.box;
  const yaw = trace.box[6];
  // rotation by -yaw
  const c = Math.cos(-yaw);
  const s = Math.sin(-yaw);

  const n = trace.x.length;
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  const intensity = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const dx = trace.x[i] - bx;
    const dy = trace.y[i] - by;
    x[i] = dx * c - dy * s;
    y[i] = dx * s + dy * c;
    z[i] = trace.z[i]; // untouched, as in ATLAS
    intensity[i] = trace.intensity[i];
  }

  const box = trace.box.slice();
  box[0] = 0;
  box[1] = 0;
  box[6] = 0;
  return { points: { x, y, z, intensity }, box };
}

/**
 * Ego-relative phantom placement: pick a trace, centre it, yaw it slightly and
 * offset it in front of the ego.
 *
 * @param {Array} traces  trace objects from io/traces.js
 * @returns {{points, box, anno}}
 */
export function placeTraceRelative(traces, {
  seed = 42,
  windowMin = 0,
  windowMax = 0,
  rng = null,
  traceIdx = null,
} = {}) {
  if (!traces || traces.length === 0) {
    throw new Error("cannot place a phantom from an empty trace library");
  }
  const r = rng || makeAtlasRng(seed);

  // draw order matches place_trace_relative
  const idx = traceIdx == null ? r.integers(traces.length) : traceIdx;
  const trace = traces[idx];
  const offX = r.uniform(RELATIVE_OFFSET.xMin, RELATIVE_OFFSET.xMax);
  const offY = r.uniform(-RELATIVE_OFFSET.yAbs, RELATIVE_OFFSET.yAbs);

  const { points: centered, box: centeredBox } = centerTrace(trace);

  const yawNew = r.uniform(-RELATIVE_OFFSET.yawAbs, RELATIVE_OFFSET.yawAbs);
  const c = Math.cos(yawNew);
  const s = Math.sin(yawNew);

  const n = centered.x.length;
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  const intensity = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const rx = centered.x[i] * c - centered.y[i] * s;
    const ry = centered.x[i] * s + centered.y[i] * c;
    x[i] = rx + offX;
    y[i] = ry + offY;
    z[i] = centered.z[i]; // original observed height
    intensity[i] = centered.intensity[i];
  }

  const box = centeredBox.slice();
  box[6] = yawNew;
  box[0] += offX;
  box[1] += offY;

  return {
    points: { x, y, z, intensity },
    box,
    anno: {
      window: [windowMin, windowMax],
      perturbation: "relative_position_fixed",
      seed,
      traceIdx: idx,
      relativePositionOffset: [offX, offY],
      yaw: yawNew,
      nTracePoints: n,
    },
  };
}
