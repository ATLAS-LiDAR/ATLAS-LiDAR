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

/**
 * Per-sector accumulation buffer.
 *
 * A spinning sensor does not show you a whole cloud at once, and it does not
 * blank when it wraps. It observes one azimuth sector at a time and each sector
 * keeps whatever was last seen there. So the display:
 *
 *   - starts EMPTY (nothing observed yet),
 *   - fills in behind the beam during the first revolution,
 *   - and from then on is always full, with every sector holding the observation
 *     from the last time the beam passed it.
 *
 * `observed[c]` is the id of the observation that currently owns azimuth column
 * c, or -1 if the beam has never been there. Ids let a driving sequence keep
 * stale keyframe data in the sectors behind the beam until it comes back round to
 * refresh them.
 */
export function makeAccumulator(sensor) {
  const observed = new Int32Array(sensor.azCellCount).fill(-1);
  return { observed, azCellCount: sensor.azCellCount };
}

/** Clear the buffer back to "nothing observed". */
export function resetAccumulator(acc) {
  acc.observed.fill(-1);
}

/** How many sectors currently hold an observation. */
export function accumulatedColumns(acc) {
  let n = 0;
  for (let c = 0; c < acc.observed.length; c++) if (acc.observed[c] >= 0) n++;
  return n;
}

/**
 * Record that the beam swept from `fromAz` to `toAz`, so those sectors now hold
 * observation `layerId`. Handles the wrap past zero.
 */
export function markSwept(acc, fromAz, toAz, layerId, sensor) {
  const n = sensor.azCellCount;
  const a = azimuthCellIndex(fromAz, sensor.azRes);
  const b = azimuthCellIndex(toAz, sensor.azRes);
  acc.observed[b] = layerId;
  if (a === b) return;
  // walk forward from a to b, wrapping; guard against a full extra lap
  let c = a;
  for (let guard = 0; guard <= n; guard++) {
    acc.observed[c] = layerId;
    if (c === b) break;
    c = c + 1 === n ? 0 : c + 1;
  }
}

/**
 * The contiguous slice of `frame.order` holding every beam of azimuth column `c`.
 * Contiguous because cellKey = azIdx * nBeams + beamIdx, so one column is one
 * block of the CSR index.
 */
export function columnRange(frame, col) {
  const nBeams = frame.sensor.nBeams;
  const lo = col * nBeams;
  return { start: frame.offsets[lo], end: frame.offsets[lo + nBeams] };
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

/**
 * Time the victim's beam spends crossing a sector of total width `widthRad`.
 * A 20 degree sector on a 10 Hz sensor is 5.6 ms.
 */
export function sectorDwellTime(sensor, widthRad) {
  return (widthRad / TWO_PI) * sensor.revolutionPeriod;
}

/** Attacker pulses that land during one crossing, at its pulse repetition rate. */
export function pulsesDuringDwell(prfHz, dwellSec) {
  return Math.floor(Math.max(0, prfHz) * Math.max(0, dwellSec));
}

/**
 * Victim (azimuth x beam) cells inside a sector - the number of distinct returns
 * an attacker would have to place to fill it.
 *
 * Comparing this against pulsesDuringDwell is the firing budget made concrete:
 * a 100 kHz attacker gets ~550 pulses while the beam crosses a 20 degree sector
 * that contains several thousand cells, so it cannot address them all.
 */
export function victimCellsInSector(sensor, azWidthRad, elWidthRad) {
  const nAz = Math.max(1, azWidthRad / sensor.azRes);
  let rows = 0;
  const half = elWidthRad / 2;
  for (let i = 0; i < sensor.nBeams; i++) {
    if (Math.abs(sensor.inclinations[i]) <= half) rows++;
  }
  return Math.round(nAz * Math.max(1, rows));
}

/** Is the victim's beam currently inside a sector of total width `widthRad`? */
export function beamInSector(sweepAz, centerAz, widthRad) {
  let d = wrapAz(sweepAz - centerAz);
  if (d > Math.PI) d -= TWO_PI;
  return Math.abs(d) <= widthRad / 2;
}
