/**
 * Polar decomposition and beam-cell quantization.
 *
 * A spinning LiDAR does not sample space continuously: it samples a discrete
 * lattice of (azimuth cell x beam) directions. Every physical constraint in this
 * sandbox is expressed on that lattice, because the hard limit on a spoofer is
 * "at most one return per cell the victim actually samples".
 *
 * Conventions match nuscenes_atlas_common.cart_2_spherical so results are
 * directly comparable to the ATLAS generation pipeline:
 *   az in [0, 2pi)   (atan2 wrapped, NOT the signed atan2 range)
 *   el = asin(z / r) (spherical elevation, not a cylindrical pitch)
 *   r  = full 3D euclidean distance from the sensor origin
 */

export const TWO_PI = Math.PI * 2;

/** Wrap an angle into [0, 2pi). */
export function wrapAz(a) {
  const m = a % TWO_PI;
  return m < 0 ? m + TWO_PI : m;
}

export function cartToSpherical(x, y, z) {
  const r = Math.sqrt(x * x + y * y + z * z);
  const az = wrapAz(Math.atan2(y, x));
  // Guard r === 0 so the origin yields el = 0 rather than NaN.
  const rSafe = r === 0 ? 1e-9 : r;
  let s = z / rSafe;
  if (s > 1) s = 1;
  else if (s < -1) s = -1;
  return { r, az, el: Math.asin(s) };
}

export function sphericalToCart(r, az, el) {
  const c = Math.cos(el);
  return [r * c * Math.cos(az), r * c * Math.sin(az), r * Math.sin(el)];
}

/**
 * Number of azimuth cells in a full revolution at resolution `dtheta` (radians).
 * Uses ceil so that an azimuth infinitesimally below 2pi still floors to an
 * index strictly inside the grid.
 */
export function azimuthCellCount(dtheta) {
  return Math.ceil(TWO_PI / dtheta);
}

/** Azimuth cell index, matching ATLAS's floor(theta % 2pi / dtheta). */
export function azimuthCellIndex(az, dtheta) {
  const idx = Math.floor(wrapAz(az) / dtheta);
  // Floating point can push an azimuth a hair under 2pi to the count itself.
  const n = azimuthCellCount(dtheta);
  return idx >= n ? n - 1 : idx;
}

/**
 * Index of the beam whose inclination is closest to `el`.
 * `inclinations` must be ascending. Binary search, verified against brute-force
 * argmin in the tests; ties resolve to the lower index to match numpy argmin.
 */
export function nearestBeamIndex(el, inclinations) {
  const n = inclinations.length;
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (inclinations[mid] < el) lo = mid + 1;
    else hi = mid;
  }
  if (lo === 0) return 0;
  if (lo === n) return n - 1;
  return Math.abs(inclinations[lo] - el) < Math.abs(el - inclinations[lo - 1]) ? lo : lo - 1;
}

/**
 * Pack (azIdx, beamIdx) into one integer key.
 *
 * ATLAS uses az_idx * 1000 + beam_idx, which silently collides once a sensor
 * has >= 1000 beams; keying on nBeams has no such ceiling for any realistic
 * grid (1086 az cells x 128 beams stays far inside int32).
 */
export function cellKey(azIdx, beamIdx, nBeams) {
  return azIdx * nBeams + beamIdx;
}

export function unpackCellKey(key, nBeams) {
  const azIdx = Math.floor(key / nBeams);
  return { azIdx, beamIdx: key - azIdx * nBeams };
}
