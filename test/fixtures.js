/** Shared fixtures for building synthetic frames on exact lattice cells. */
import { PointFrame } from "../src/core/frame.js";
import { sphericalToCart } from "../src/core/polar.js";

export function xyzi(tuples) {
  const a = new Float32Array(tuples.length * 4);
  tuples.forEach((t, k) => {
    a[k * 4] = t[0];
    a[k * 4 + 1] = t[1];
    a[k * 4 + 2] = t[2];
    a[k * 4 + 3] = t[3] ?? 0;
  });
  return a;
}

/** A point at the centre of azimuth cell `azIdx` on beam `beamIdx`, at range r. */
export function atCell(sensor, azIdx, beamIdx, r, intensity = 1) {
  const az = (azIdx + 0.5) * sensor.azRes;
  const el = sensor.inclinations[beamIdx];
  const [x, y, z] = sphericalToCart(r, az, el);
  return [x, y, z, intensity];
}

/** A point at an exact (azimuth, elevation) direction and range. */
export function atDir(azRad, elRad, r, intensity = 1) {
  const [x, y, z] = sphericalToCart(r, azRad, elRad);
  return [x, y, z, intensity];
}

export function frameOf(sensor, tuples, meta = {}) {
  return PointFrame.fromXYZI(xyzi(tuples), sensor, meta);
}

/** An empty frame, for testing free-space behaviour. */
export function emptyFrame(sensor) {
  return PointFrame.fromXYZI(new Float32Array(0), sensor);
}

/**
 * A wall: every beam filled at constant range across an azimuth span,
 * approximating a surface that occludes everything behind it.
 */
export function wallFrame(sensor, { azFromDeg, azToDeg, range }) {
  const tuples = [];
  const lo = Math.floor((azFromDeg * Math.PI) / 180 / sensor.azRes);
  const hi = Math.ceil((azToDeg * Math.PI) / 180 / sensor.azRes);
  for (let a = lo; a <= hi; a++) {
    for (let b = 0; b < sensor.nBeams; b++) {
      tuples.push(atCell(sensor, ((a % sensor.azCellCount) + sensor.azCellCount) % sensor.azCellCount, b, range));
    }
  }
  return frameOf(sensor, tuples);
}
