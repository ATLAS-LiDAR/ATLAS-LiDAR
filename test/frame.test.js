import { describe, it, expect } from "./harness.js";
import { rng, uniform } from "./util.js";
import { PointFrame } from "../src/core/frame.js";
import { configureSensor } from "../src/core/sensor.js";
import { cartToSpherical, cellKey } from "../src/core/polar.js";
import { sphericalToCart } from "../src/core/polar.js";

const sensor = configureSensor("hdl32e");

/** Build an interleaved xyzi buffer from [x,y,z,i] tuples. */
function xyzi(tuples) {
  const a = new Float32Array(tuples.length * 4);
  tuples.forEach((t, k) => {
    a[k * 4] = t[0];
    a[k * 4 + 1] = t[1];
    a[k * 4 + 2] = t[2];
    a[k * 4 + 3] = t[3] ?? 0;
  });
  return a;
}

/** A point placed exactly on a given lattice cell at a given range. */
function atCell(azIdx, beamIdx, r, s = sensor) {
  const az = (azIdx + 0.5) * s.azRes;
  const el = s.inclinations[beamIdx];
  const [x, y, z] = sphericalToCart(r, az, el);
  return [x, y, z, 1];
}

describe("PointFrame construction", () => {
  it("stores the cloud and derives polar coordinates per point", () => {
    const f = PointFrame.fromXYZI(xyzi([[3, 4, 0, 0.5], [0, 0, 10, 0.25]]), sensor);
    expect(f.count).toBe(2);
    expect(f.r[0]).toBeCloseTo(5, 1e-5);
    expect(f.intensity[0]).toBeCloseTo(0.5, 1e-6);
    expect(f.az[0]).toBeCloseTo(cartToSpherical(3, 4, 0).az, 1e-6);
    expect(f.el[1]).toBeCloseTo(Math.PI / 2, 1e-6);
  });

  it("assigns each point to the lattice cell it was constructed on", () => {
    const f = PointFrame.fromXYZI(xyzi([atCell(100, 7, 20), atCell(900, 31, 55)]), sensor);
    expect(f.azIdx[0]).toBe(100);
    expect(f.beamIdx[0]).toBe(7);
    expect(f.azIdx[1]).toBe(900);
    expect(f.beamIdx[1]).toBe(31);
    expect(f.cell[0]).toBe(cellKey(100, 7, sensor.nBeams));
  });

  it("rejects a buffer whose length is not a multiple of the stride", () => {
    expect(() => PointFrame.fromXYZI(new Float32Array(7), sensor)).toThrow("stride");
  });

  it("accepts an empty cloud without throwing", () => {
    const f = PointFrame.fromXYZI(new Float32Array(0), sensor);
    expect(f.count).toBe(0);
    expect(f.occupiedCells).toBe(0);
    expect(f.nearestInCell(0)).toBe(-1);
  });
});

describe("cell occupancy index", () => {
  it("groups points by cell with range ascending inside each cell", () => {
    // deliberately out of order, and two cells interleaved
    const f = PointFrame.fromXYZI(
      xyzi([
        atCell(10, 3, 30),
        atCell(50, 8, 12),
        atCell(10, 3, 8),
        atCell(10, 3, 19),
        atCell(50, 8, 40),
      ]),
      sensor
    );

    const kA = cellKey(10, 3, sensor.nBeams);
    const a = f.cellRange(kA);
    expect(a.end - a.start).toBe(3);
    const ranges = [];
    for (let i = a.start; i < a.end; i++) ranges.push(f.r[f.order[i]]);
    expect(ranges).toBeCloseToArray([8, 19, 30], 1e-3);

    const kB = cellKey(50, 8, sensor.nBeams);
    const b = f.cellRange(kB);
    expect(b.end - b.start).toBe(2);
    expect(f.r[f.order[b.start]]).toBeCloseTo(12, 1e-3);
  });

  it("reports an empty range for an unoccupied cell", () => {
    const f = PointFrame.fromXYZI(xyzi([atCell(10, 3, 30)]), sensor);
    const k = cellKey(11, 3, sensor.nBeams);
    const s = f.cellRange(k);
    expect(s.end - s.start).toBe(0);
    expect(f.nearestInCell(k)).toBe(-1);
  });

  it("nearestInCell returns the closest return, which is what occludes", () => {
    const f = PointFrame.fromXYZI(
      xyzi([atCell(10, 3, 30), atCell(10, 3, 8), atCell(10, 3, 19)]),
      sensor
    );
    const k = cellKey(10, 3, sensor.nBeams);
    const idx = f.nearestInCell(k);
    expect(f.r[idx]).toBeCloseTo(8, 1e-3);
  });

  it("counts occupied cells, not points", () => {
    const f = PointFrame.fromXYZI(
      xyzi([atCell(10, 3, 30), atCell(10, 3, 8), atCell(50, 8, 12)]),
      sensor
    );
    expect(f.count).toBe(3);
    expect(f.occupiedCells).toBe(2);
  });

  it("places every point exactly once in the order array", () => {
    const next = rng(99);
    const pts = [];
    for (let i = 0; i < 4000; i++) {
      pts.push([uniform(next, -60, 60), uniform(next, -60, 60), uniform(next, -4, 8), 0.3]);
    }
    const f = PointFrame.fromXYZI(xyzi(pts), sensor);
    expect(f.order).toHaveLength(f.count);
    const seen = new Uint8Array(f.count);
    for (let i = 0; i < f.order.length; i++) {
      const p = f.order[i];
      expect(seen[p]).toBe(0);
      seen[p] = 1;
    }
    for (let i = 0; i < f.count; i++) expect(seen[i]).toBe(1);
  });

  it("keeps each point inside the cell slice its own key points at", () => {
    const next = rng(1234);
    const pts = [];
    for (let i = 0; i < 3000; i++) {
      pts.push([uniform(next, -70, 70), uniform(next, -70, 70), uniform(next, -3, 6), 0]);
    }
    const f = PointFrame.fromXYZI(xyzi(pts), sensor);
    for (let p = 0; p < f.count; p += 7) {
      const { start, end } = f.cellRange(f.cell[p]);
      let found = false;
      for (let i = start; i < end; i++) if (f.order[i] === p) found = true;
      expect(found).toBeTruthy();
    }
  });

  it("sorts by range within every cell of a random cloud", () => {
    const next = rng(555);
    const pts = [];
    for (let i = 0; i < 5000; i++) {
      // concentrate into few cells so multi-return cells actually occur
      const az = uniform(next, 0, 0.2);
      const el = sensor.inclinations[Math.floor(uniform(next, 0, sensor.nBeams))];
      const [x, y, z] = sphericalToCart(uniform(next, 2, 90), az, el);
      pts.push([x, y, z, 0]);
    }
    const f = PointFrame.fromXYZI(xyzi(pts), sensor);
    let multi = 0;
    for (const k of f.keys()) {
      const { start, end } = f.cellRange(k);
      if (end - start > 1) multi++;
      for (let i = start + 1; i < end; i++) {
        expect(f.r[f.order[i]]).toBeGreaterThanOrEqual(f.r[f.order[i - 1]]);
      }
    }
    expect(multi).toBeGreaterThan(0);
  });
});

describe("performance", () => {
  it("indexes a full-size 120k-point frame well inside an interactive budget", () => {
    // Guards against an accidental O(n^2) or per-point Map allocation; the UI
    // rebuilds this on every parameter change.
    const next = rng(2024);
    const n = 120000;
    const buf = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      buf[i * 4] = uniform(next, -70, 70);
      buf[i * 4 + 1] = uniform(next, -70, 70);
      buf[i * 4 + 2] = uniform(next, -3, 8);
      buf[i * 4 + 3] = uniform(next, 0, 255);
    }
    const t0 = performance.now();
    const f = PointFrame.fromXYZI(buf, sensor);
    const dt = performance.now() - t0;
    expect(f.count).toBe(n);
    expect(dt).toBeLessThan(1500);
  });
});
