/**
 * Sweep state for animating the victim's rotation over a recorded frame.
 *
 * A recorded cloud has already been sampled, so there are no rays left to cast.
 * But we recovered the scan lattice, and every point carries the azimuth column
 * it was sampled in - so the spin can be animated *faithfully* rather than
 * decoratively: reveal points in true azimuth order, which is the order the
 * sensor actually acquired them.
 *
 * This is cheap because PointFrame's CSR index is already azimuth-sorted:
 * cellKey = azIdx * nBeams + beamIdx is monotonic in azIdx, so all the points of
 * columns 0..k-1 occupy a prefix of `frame.order`, and the prefix length is one
 * offsets lookup. No per-frame scan, no re-sorting.
 *
 * Animating the sweep also makes the synchronization model visible: the attacker
 * can only act while the victim's beam is inside its sector, and `beamInSector`
 * is what a renderer keys that off.
 */
import { TWO_PI, wrapAz, azimuthCellIndex } from "./polar.js";

/**
 * Number of points the beam has already passed, for a sweep at azimuth column
 * `azIdx`. O(1).
 */
export function sweptPointCount(frame, azIdx) {
  const nCells = frame.sensor.azCellCount;
  if (azIdx <= 0) return 0;
  if (azIdx >= nCells) return frame.count;
  return frame.offsets[azIdx * frame.sensor.nBeams];
}

/** Sensor column index for a sweep azimuth. */
export function sweepAzToCell(azRad, sensor) {
  return azimuthCellIndex(azRad, sensor.azRes);
}

/**
 * Advance the sweep azimuth by `dtSec` of wall time.
 * `speed` is a display multiplier: 1 means real time at the sensor's spin rate.
 */
export function advanceSweep(azRad, dtSec, sensor, speed = 1) {
  return wrapAz(azRad + dtSec * TWO_PI * sensor.hz * speed);
}

/** Revolutions completed in `seconds` of wall time. */
export function sweepRevolutions(seconds, sensor, speed = 1) {
  return seconds * sensor.hz * speed;
}

/** Is the victim's beam currently inside a sector of total width `widthRad`? */
export function beamInSector(sweepAz, centerAz, widthRad) {
  let d = wrapAz(sweepAz - centerAz);
  if (d > Math.PI) d -= TWO_PI;
  return Math.abs(d) <= widthRad / 2;
}
