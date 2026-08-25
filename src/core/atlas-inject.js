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
 * Both placement policies are ported:
 *   place_trace_relative - ego-fixed. The phantom sits 15-20 m ahead of the ego
 *                          in EVERY frame, so it moves with the vehicle.
 *   place_trace_global   - world-fixed. The phantom is placed once in the anchor
 *                          frame, lifted into world coordinates, and projected
 *                          back down into each later frame, so the ego drives
 *                          past it.
 * The difference is only visible in a sequence, and it is the difference between
 * a phantom that follows you and an obstacle that is really there.
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

import { applyPose, invertRigid4, poseYaw } from "./pose.js";

/**
 * Ego-relative phantom placement: pick a trace, centre it, yaw it slightly and
 * offset it in front of the ego.
 *
 * ATLAS samples the trace, the offset and the yaw at random. `traceIdx`,
 * `offsetXY` and `yaw` let a user choose them instead. That keeps the TRANSFORM
 * exactly ATLAS's - de-rotate, centre, re-yaw, translate, z untouched - and only
 * replaces where the sampled parameters come from, so the phantom is still built
 * the way ATLAS builds one. `anno.placement` records which it was.
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
  offsetXY = null,
  yaw = null,
} = {}) {
  if (!traces || traces.length === 0) {
    throw new Error("cannot place a phantom from an empty trace library");
  }
  const r = rng || makeAtlasRng(seed);

  // Draw order matches place_trace_relative. The draws happen even when a value
  // is overridden, so a given seed keeps producing the same *other* values.
  const drawnIdx = r.integers(traces.length);
  const drawnX = r.uniform(RELATIVE_OFFSET.xMin, RELATIVE_OFFSET.xMax);
  const drawnY = r.uniform(-RELATIVE_OFFSET.yAbs, RELATIVE_OFFSET.yAbs);

  const idx = traceIdx == null ? drawnIdx : Math.max(0, Math.min(traces.length - 1, traceIdx));
  const trace = traces[idx];
  const offX = offsetXY ? offsetXY[0] : drawnX;
  const offY = offsetXY ? offsetXY[1] : drawnY;

  const { points: centered, box: centeredBox } = centerTrace(trace);

  const drawnYaw = r.uniform(-RELATIVE_OFFSET.yawAbs, RELATIVE_OFFSET.yawAbs);
  const yawNew = yaw == null ? drawnYaw : yaw;
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
      placement: offsetXY || yaw != null || traceIdx != null ? "user" : "atlas-random",
      traceRange: trace.range ?? null,
    },
  };
}

/**
 * World-fixed phantom placement: place it relative to the anchor frame's ego,
 * then lift the points and the box into world coordinates.
 *
 * The box yaw gains the anchor's heading, as in ATLAS
 * (`box[6] + arctan2(pose[1,0], pose[0,0])`).
 */
export function placeTraceGlobal(traces, {
  seed = 42,
  windowMin = 0,
  windowMax = 0,
  rng = null,
  traceIdx = null,
  offsetXY = null,
  yaw = null,
  anchorPose = null,
} = {}) {
  if (!anchorPose) throw new Error("placeTraceGlobal needs an anchorPose (ego->global)");

  // identical to the relative placement, then lifted
  const rel = placeTraceRelative(traces, {
    seed, windowMin, windowMax, rng, traceIdx, offsetXY, yaw,
  });

  const points = {
    x: new Float64Array(rel.points.x.length),
    y: new Float64Array(rel.points.y.length),
    z: new Float64Array(rel.points.z.length),
    intensity: rel.points.intensity,
  };
  for (let i = 0; i < rel.points.x.length; i++) {
    const p = applyPose(anchorPose, [rel.points.x[i], rel.points.y[i], rel.points.z[i]]);
    points.x[i] = p[0];
    points.y[i] = p[1];
    points.z[i] = p[2];
  }

  const box = rel.box.slice();
  const c = applyPose(anchorPose, [rel.box[0], rel.box[1], rel.box[2]]);
  box[0] = c[0];
  box[1] = c[1];
  box[2] = c[2];
  box[6] = rel.box[6] + poseYaw(anchorPose);

  return {
    points,
    box,
    anno: { ...rel.anno, perturbation: "global_position_fixed" },
  };
}

/**
 * Bring a world-fixed phantom down into a particular frame's ego coordinates.
 * Port of ATLAS's transform_spoof_to_ego_frame.
 */
export function transformSpoofToEgoFrame(pointsGlobal, boxGlobal, targetPose) {
  const inv = invertRigid4(targetPose);
  const n = pointsGlobal.x.length;
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = applyPose(inv, [pointsGlobal.x[i], pointsGlobal.y[i], pointsGlobal.z[i]]);
    x[i] = p[0];
    y[i] = p[1];
    z[i] = p[2];
  }

  const box = boxGlobal.slice();
  const c = applyPose(inv, [boxGlobal[0], boxGlobal[1], boxGlobal[2]]);
  box[0] = c[0];
  box[1] = c[1];
  box[2] = c[2];
  box[6] = boxGlobal[6] - poseYaw(targetPose);

  return { points: { x, y, z, intensity: pointsGlobal.intensity }, box };
}
