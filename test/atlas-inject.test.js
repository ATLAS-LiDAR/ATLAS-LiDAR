import { describe, it, expect } from "./harness.js";
import {
  parseTraceLibrary,
  loadTraceLibrary,
  loadTraceManifest,
  getTrace,
} from "../src/io/traces.js";
import {
  centerTrace,
  placeTraceRelative,
  makeAtlasRng,
  RELATIVE_OFFSET,
} from "../src/core/atlas-inject.js";

describe("trace library loading", () => {
  it("lists the exported libraries", async () => {
    const m = await loadTraceManifest("../assets/traces/manifest.json");
    expect(m.length).toBe(6);
    const diffs = new Set(m.map((e) => e.difficulty));
    for (const d of ["easy", "medium", "hard"]) expect([...diffs]).toContain(d);
  });

  it("loads a real library with per-trace boxes and counts", async () => {
    const lib = await loadTraceLibrary("../assets/traces/nuscenes-medium.atlastrace");
    expect(lib.header.dataset).toBe("nuscenes");
    expect(lib.header.difficulty).toBe("medium");
    expect(lib.header.nTraces).toBe(200);
    expect(lib.header.counts).toHaveLength(200);
    expect(lib.header.boxes).toHaveLength(200);
    expect(lib.header.counts.reduce((a, b) => a + b, 0)).toBe(lib.header.totalPoints);
  });

  it("rejects a buffer that is not a trace library", () => {
    expect(() => parseTraceLibrary(new Uint8Array(64).buffer)).toThrow("magic");
  });

  it("slices out an individual trace with its box", async () => {
    const lib = await loadTraceLibrary("../assets/traces/nuscenes-medium.atlastrace");
    const t = getTrace(lib, 0);
    expect(t.x).toHaveLength(lib.header.counts[0]);
    expect(t.box).toHaveLength(7);
    expect(t.intensity.length).toBe(t.x.length);
    for (let i = 0; i < t.x.length; i++) expect(Number.isFinite(t.x[i])).toBeTruthy();
  });

  it("reproduces the measured density ordering easy > medium > hard", async () => {
    // This IS the meaning of the inject_*_{easy,medium,hard} split names.
    const med = {};
    for (const d of ["easy", "medium", "hard"]) {
      const lib = await loadTraceLibrary(`../assets/traces/nuscenes-${d}.atlastrace`);
      const c = [...lib.header.counts].sort((a, b) => a - b);
      med[d] = c[c.length >> 1];
    }
    expect(med.easy).toBeGreaterThan(med.medium);
    expect(med.medium).toBeGreaterThan(med.hard);
    expect(med.hard).toBeLessThan(15);
  });

  it("keeps points near their own box, so the box really describes the trace", async () => {
    const lib = await loadTraceLibrary("../assets/traces/nuscenes-medium.atlastrace");
    for (const k of [0, 7, 50, 199]) {
      const t = getTrace(lib, k);
      const [bx, by] = t.box;
      const diag = Math.hypot(t.box[3], t.box[4]);
      for (let i = 0; i < t.x.length; i++) {
        expect(Math.hypot(t.x[i] - bx, t.y[i] - by)).toBeLessThan(diag + 2.0);
      }
    }
  });
});

describe("centerTrace — port of ATLAS center_trace", () => {
  /** A synthetic trace: a box at (10, 0) yawed 90 deg, with two points on it. */
  function synth() {
    return {
      // points sit 1 m ahead and 1 m left of the box centre in world axes
      x: Float64Array.from([11, 10]),
      y: Float64Array.from([0, 1]),
      z: Float64Array.from([-1.2, -1.0]),
      intensity: Float32Array.from([10, 20]),
      box: [10, 0, -1.0, 4, 2, 1.6, Math.PI / 2],
    };
  }

  it("moves the box to the origin and zeroes its yaw", () => {
    const { points, box } = centerTrace(synth());
    expect(box[0]).toBeCloseTo(0, 1e-12);
    expect(box[1]).toBeCloseTo(0, 1e-12);
    expect(box[6]).toBeCloseTo(0, 1e-12);
    // dimensions are untouched
    expect(box[3]).toBeCloseTo(4, 1e-12);
    expect(box[4]).toBeCloseTo(2, 1e-12);
    expect(points.x).toHaveLength(2);
  });

  it("de-rotates the points by the box yaw", () => {
    // point 0 is at +1 in x relative to the centre; de-rotating by 90 deg sends
    // it to -1 in y (rotation by -yaw)
    const { points } = centerTrace(synth());
    expect(points.x[0]).toBeCloseTo(0, 1e-9);
    expect(points.y[0]).toBeCloseTo(-1, 1e-9);
  });

  it("leaves z ALONE, as ATLAS does", () => {
    // center_trace only touches xy; z stays at its originally observed height and
    // is corrected later by ground anchoring. Recentring z here would double-
    // correct and sink the phantom.
    const t = synth();
    const { points, box } = centerTrace(t);
    expect(points.z[0]).toBeCloseTo(t.z[0], 1e-12);
    expect(points.z[1]).toBeCloseTo(t.z[1], 1e-12);
    expect(box[2]).toBeCloseTo(t.box[2], 1e-12);
  });

  it("preserves intensity", () => {
    const { points } = centerTrace(synth());
    expect(points.intensity[0]).toBeCloseTo(10, 1e-6);
  });
});

describe("placeTraceRelative — port of ATLAS place_trace_relative", () => {
  function libStub() {
    // three traces, each a small cluster around its own box
    const mk = (cx, cy, yaw) => ({
      x: Float64Array.from([cx + 1, cx, cx - 1]),
      y: Float64Array.from([cy, cy + 0.5, cy]),
      z: Float64Array.from([-1.4, -1.2, -1.0]),
      intensity: Float32Array.from([5, 10, 15]),
      box: [cx, cy, -1.2, 4.2, 1.8, 1.6, yaw],
    });
    return [mk(20, 3, 0.4), mk(-15, 8, -1.1), mk(30, -12, 2.0)];
  }

  it("offsets the phantom into ATLAS's placement window", () => {
    // uniform(15, 20) ahead and uniform(-3, 3) lateral
    expect(RELATIVE_OFFSET.xMin).toBe(15);
    expect(RELATIVE_OFFSET.xMax).toBe(20);
    expect(RELATIVE_OFFSET.yAbs).toBe(3);
    for (let seed = 1; seed <= 40; seed++) {
      const { box } = placeTraceRelative(libStub(), { seed });
      expect(box[0]).toBeGreaterThanOrEqual(15);
      expect(box[0]).toBeLessThanOrEqual(20);
      expect(Math.abs(box[1])).toBeLessThanOrEqual(3);
    }
  });

  it("yaws the phantom within +-0.2 rad", () => {
    for (let seed = 1; seed <= 40; seed++) {
      const { box } = placeTraceRelative(libStub(), { seed });
      expect(Math.abs(box[6])).toBeLessThanOrEqual(0.2);
    }
  });

  it("keeps every point of the chosen trace", () => {
    const lib = libStub();
    const { points, anno } = placeTraceRelative(lib, { seed: 3 });
    expect(points.x).toHaveLength(lib[anno.traceIdx].x.length);
  });

  it("places the points consistently with the returned box", () => {
    const { points, box } = placeTraceRelative(libStub(), { seed: 11 });
    const diag = Math.hypot(box[3], box[4]);
    for (let i = 0; i < points.x.length; i++) {
      expect(Math.hypot(points.x[i] - box[0], points.y[i] - box[1])).toBeLessThan(diag);
    }
  });

  it("carries the original observed z through untouched", () => {
    const lib = libStub();
    const { points, anno } = placeTraceRelative(lib, { seed: 5 });
    const src = lib[anno.traceIdx];
    for (let i = 0; i < points.z.length; i++) {
      expect(points.z[i]).toBeCloseTo(src.z[i], 1e-9);
    }
  });

  it("is reproducible for a seed and varies across seeds", () => {
    const a = placeTraceRelative(libStub(), { seed: 42 });
    const b = placeTraceRelative(libStub(), { seed: 42 });
    const c = placeTraceRelative(libStub(), { seed: 43 });
    expect(a.box).toBeCloseToArray(b.box, 1e-12);
    expect(a.box[0] === c.box[0] && a.box[1] === c.box[1]).toBeFalsy();
  });

  it("records the annotation fields ATLAS stores", () => {
    const { anno } = placeTraceRelative(libStub(), { seed: 8, windowMin: 2, windowMax: 9 });
    expect(anno.perturbation).toBe("relative_position_fixed");
    expect(anno.window).toEqual([2, 9]);
    expect(typeof anno.traceIdx).toBe("number");
    expect(anno.relativePositionOffset).toHaveLength(2);
    expect(typeof anno.yaw).toBe("number");
  });

  it("works on a real exported library", async () => {
    const lib = await loadTraceLibrary("../assets/traces/nuscenes-hard.atlastrace");
    const traces = [];
    for (let i = 0; i < lib.header.nTraces; i++) traces.push(getTrace(lib, i));
    const { points, box, anno } = placeTraceRelative(traces, { seed: 21 });
    expect(points.x.length).toBeGreaterThan(0);
    expect(box[0]).toBeGreaterThanOrEqual(15);
    expect(anno.traceIdx).toBeGreaterThanOrEqual(0);
    // a hard trace is sparse by construction
    expect(points.x.length).toBeLessThan(20);
  });

  it("throws on an empty library rather than injecting nothing silently", () => {
    expect(() => placeTraceRelative([], { seed: 1 })).toThrow("empty");
  });
});

describe("makeAtlasRng", () => {
  it("gives uniform draws in range and is seed-reproducible", () => {
    const r = makeAtlasRng(9);
    for (let i = 0; i < 100; i++) {
      const v = r.uniform(15, 20);
      expect(v).toBeGreaterThanOrEqual(15);
      expect(v).toBeLessThan(20);
    }
    expect(makeAtlasRng(4).uniform(0, 1)).toBeCloseTo(makeAtlasRng(4).uniform(0, 1), 1e-15);
  });

  it("draws integers in [0, n)", () => {
    const r = makeAtlasRng(2);
    for (let i = 0; i < 200; i++) {
      const k = r.integers(7);
      expect(k).toBeGreaterThanOrEqual(0);
      expect(k).toBeLessThan(7);
      expect(Number.isInteger(k)).toBeTruthy();
    }
  });
});
