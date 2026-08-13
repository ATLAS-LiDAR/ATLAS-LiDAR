/**
 * PointFrame: a recorded LiDAR frame indexed onto the sensor's scan lattice.
 *
 * Every physical constraint on a spoofer is expressed per (azimuth cell x beam)
 * cell, so a recorded cloud has to be projected back onto that lattice before
 * any attack can be reasoned about. The index is CSR-style (counting sort into
 * offsets) rather than a Map, because the UI rebuilds it on every parameter
 * change over 30k-120k points and per-point object allocation would dominate.
 *
 * Points inside each cell are sorted by ascending range, which is the order the
 * occlusion logic needs: order[cellRange(k).start] is the return that a real
 * sensor would report first.
 */
import { cartToSpherical, azimuthCellIndex, nearestBeamIndex, cellKey } from "./polar.js";

/** Insertion sort is the right choice: cell occupancy is almost always 1-3. */
function sortSliceByRange(order, start, end, r) {
  for (let i = start + 1; i < end; i++) {
    const v = order[i];
    const key = r[v];
    let j = i - 1;
    while (j >= start && r[order[j]] > key) {
      order[j + 1] = order[j];
      j--;
    }
    order[j + 1] = v;
  }
}

export class PointFrame {
  /**
   * @param {object} arrays deinterleaved cloud + sensor
   */
  constructor({ x, y, z, intensity, time, sensor, meta = {} }) {
    const n = x.length;
    this.count = n;
    this.sensor = sensor;
    this.meta = meta;
    this.x = x;
    this.y = y;
    this.z = z;
    this.intensity = intensity;
    this.time = time || null;

    const nBeams = sensor.nBeams;
    const nCells = sensor.azCellCount * nBeams;
    this.nCells = nCells;

    this.r = new Float64Array(n);
    this.az = new Float64Array(n);
    this.el = new Float64Array(n);
    this.azIdx = new Int32Array(n);
    this.beamIdx = new Int32Array(n);
    this.cell = new Int32Array(n);

    const counts = new Int32Array(nCells + 1);
    const inc = sensor.inclinations;
    const azResLocal = sensor.azRes;

    for (let i = 0; i < n; i++) {
      const s = cartToSpherical(x[i], y[i], z[i]);
      this.r[i] = s.r;
      this.az[i] = s.az;
      this.el[i] = s.el;
      const a = azimuthCellIndex(s.az, azResLocal);
      const b = nearestBeamIndex(s.el, inc);
      this.azIdx[i] = a;
      this.beamIdx[i] = b;
      const k = cellKey(a, b, nBeams);
      this.cell[i] = k;
      counts[k + 1]++;
    }

    // prefix sum -> CSR offsets
    let occupied = 0;
    for (let c = 0; c < nCells; c++) {
      if (counts[c + 1] > 0) occupied++;
      counts[c + 1] += counts[c];
    }
    this.offsets = counts;
    this.occupiedCells = occupied;

    // scatter point indices into their cell slots
    const cursor = new Int32Array(nCells);
    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const k = this.cell[i];
      order[counts[k] + cursor[k]] = i;
      cursor[k]++;
    }
    this.order = order;

    // range-sort within each occupied cell
    for (let i = 0; i < n; ) {
      const k = this.cell[order[i]];
      const start = counts[k];
      const end = counts[k + 1];
      if (end - start > 1) sortSliceByRange(order, start, end, this.r);
      i = end > i ? end : i + 1;
    }
  }

  /**
   * Build from an interleaved buffer. stride 4 is KITTI's x,y,z,intensity;
   * stride 5 is nuScenes' x,y,z,intensity,time; stride 3 is bare xyz.
   */
  static fromInterleaved(buf, sensor, stride = 4, meta = {}) {
    if (buf.length % stride !== 0) {
      throw new Error(
        "buffer length " + buf.length + " is not a multiple of stride " + stride
      );
    }
    const n = buf.length / stride;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const z = new Float32Array(n);
    const intensity = new Float32Array(n);
    const time = stride >= 5 ? new Float32Array(n) : null;
    for (let i = 0; i < n; i++) {
      const o = i * stride;
      x[i] = buf[o];
      y[i] = buf[o + 1];
      z[i] = buf[o + 2];
      if (stride >= 4) intensity[i] = buf[o + 3];
      if (time) time[i] = buf[o + 4];
    }
    return new PointFrame({ x, y, z, intensity, time, sensor, meta });
  }

  static fromXYZI(buf, sensor, meta = {}) {
    return PointFrame.fromInterleaved(buf, sensor, 4, meta);
  }

  /** CSR slice for a cell key; empty (start === end) if unoccupied. */
  cellRange(key) {
    if (key < 0 || key >= this.nCells) return { start: 0, end: 0 };
    return { start: this.offsets[key], end: this.offsets[key + 1] };
  }

  /** Point index of the closest return in a cell, or -1 if the cell is empty. */
  nearestInCell(key) {
    if (key < 0 || key >= this.nCells) return -1;
    const s = this.offsets[key];
    return this.offsets[key + 1] > s ? this.order[s] : -1;
  }

  /** Occupied cell keys, ascending. */
  keys() {
    const out = new Int32Array(this.occupiedCells);
    let m = 0;
    for (let c = 0; c < this.nCells; c++) {
      if (this.offsets[c + 1] > this.offsets[c]) out[m++] = c;
    }
    return out;
  }
}
