/**
 * Rigid 4x4 pose maths, for driving sequences.
 *
 * Poses are flat 16-element row-major arrays, as written by
 * tools/export_sequence.py, and mean **ego -> global**: the transform that lifts
 * a point in that keyframe's sensor coordinates into the world.
 *
 * ATLAS composes them the same way. `place_trace_global` lifts a phantom into
 * the world with the anchor frame's pose, and
 * `transform_spoof_to_ego_frame` brings it back down into each later frame with
 * that frame's inverse - which is what makes a world-fixed phantom appear to
 * recede as the ego drives past it. Yaw is read as atan2(m[1][0], m[0][0]),
 * matching ATLAS exactly.
 */

export const IDENTITY4 = Object.freeze([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** Row-major 4x4 product: matMul4(a, b) means "apply b, then a". */
export function matMul4(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = s;
    }
  }
  return out;
}

/**
 * Inverse of a rigid transform, using R^-1 = R^T rather than a general solve.
 * Exact and cheap; these are all rotation + translation.
 */
export function invertRigid4(m) {
  const r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const t = [m[3], m[7], m[11]];
  // R^T
  const rt = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
  // -R^T t
  const tx = -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]);
  const ty = -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]);
  const tz = -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]);
  return [rt[0], rt[1], rt[2], tx, rt[3], rt[4], rt[5], ty, rt[6], rt[7], rt[8], tz, 0, 0, 0, 1];
}

/** R p + t. */
export function applyPose(m, p) {
  return [
    m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
    m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
    m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
  ];
}

/** Heading of a pose, as ATLAS reads it. */
export function poseYaw(m) {
  return Math.atan2(m[4], m[0]);
}

/** Transform a whole {x,y,z,...} point set, preserving any intensity array. */
export function transformPoints(m, pts) {
  const n = pts.x.length;
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const px = pts.x[i];
    const py = pts.y[i];
    const pz = pts.z[i];
    x[i] = m[0] * px + m[1] * py + m[2] * pz + m[3];
    y[i] = m[4] * px + m[5] * py + m[6] * pz + m[7];
    z[i] = m[8] * px + m[9] * py + m[10] * pz + m[11];
  }
  return pts.intensity ? { x, y, z, intensity: pts.intensity } : { x, y, z };
}

/** Transform carrying coordinates from frame A's ego into frame B's ego. */
export function relativePose(poseA, poseB) {
  return matMul4(invertRigid4(poseB), poseA);
}
