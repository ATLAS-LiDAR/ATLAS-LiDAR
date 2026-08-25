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
  placeTraceGlobal,
  transformSpoofToEgoFrame,
  makeAtlasRng,
  RELATIVE_OFFSET,
} from "../src/core/atlas-inject.js";
import { applyPose, invertRigid4, poseYaw } from "../src/core/pose.js";

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

  // --- user-chosen trace and location -------------------------------------
  // ATLAS samples these at random. Letting the user pick them keeps the
  // TRANSFORM identical (centre, yaw, offset) and only replaces the sampled
  // parameters, so the phantom is still built exactly the way ATLAS builds one.

  it("uses an explicitly chosen trace", () => {
    for (const k of [0, 1, 2]) {
      expect(placeTraceRelative(libStub(), { seed: 1, traceIdx: k }).anno.traceIdx).toBe(k);
    }
  });

  it("places the phantom at an explicitly chosen location", () => {
    const { box, anno } = placeTraceRelative(libStub(), { seed: 1, offsetXY: [42, -7.5] });
    expect(box[0]).toBeCloseTo(42, 1e-9);
    expect(box[1]).toBeCloseTo(-7.5, 1e-9);
    expect(anno.relativePositionOffset).toBeCloseToArray([42, -7.5], 1e-9);
    expect(anno.placement).toBe("user");
  });

  it("honours a chosen location outside ATLAS's random window", () => {
    // the window is only where ATLAS samples; the transform works anywhere
    const { box } = placeTraceRelative(libStub(), { seed: 1, offsetXY: [-30, 12] });
    expect(box[0]).toBeCloseTo(-30, 1e-9);
    expect(box[1]).toBeCloseTo(12, 1e-9);
  });

  it("uses an explicitly chosen yaw", () => {
    const { box } = placeTraceRelative(libStub(), { seed: 1, yaw: 1.0 });
    expect(box[6]).toBeCloseTo(1.0, 1e-9);
  });

  it("still moves the points with the chosen box", () => {
    const { points, box } = placeTraceRelative(libStub(), {
      seed: 1, traceIdx: 0, offsetXY: [25, 4], yaw: 0.5,
    });
    const diag = Math.hypot(box[3], box[4]);
    for (let i = 0; i < points.x.length; i++) {
      expect(Math.hypot(points.x[i] - box[0], points.y[i] - box[1])).toBeLessThan(diag);
    }
  });

  it("marks a random placement as such, so the readout can say which it was", () => {
    expect(placeTraceRelative(libStub(), { seed: 1 }).anno.placement).toBe("atlas-random");
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

describe("placeTraceGlobal — port of ATLAS place_trace_global", () => {
  function libStub() {
    const mk = (cx, cy, yaw) => ({
      x: Float64Array.from([cx + 1, cx, cx - 1]),
      y: Float64Array.from([cy, cy + 0.5, cy]),
      z: Float64Array.from([-1.4, -1.2, -1.0]),
      intensity: Float32Array.from([5, 10, 15]),
      box: [cx, cy, -1.2, 4.2, 1.8, 1.6, yaw],
    });
    return [mk(20, 3, 0.4), mk(-15, 8, -1.1), mk(30, -12, 2.0)];
  }
  /** ego->global: rotation about z then translation. */
  const pose = (yaw, tx, ty) => {
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    return [c, -s, 0, tx, s, c, 0, ty, 0, 0, 1, 0, 0, 0, 0, 1];
  };

  it("lifts the phantom into world coordinates using the anchor pose", () => {
    const anchor = pose(0.7, 100, -50);
    const g = placeTraceGlobal(libStub(), { seed: 5, anchorPose: anchor });
    // the box is far from the origin because it now lives in the world frame
    expect(Math.hypot(g.box[0], g.box[1])).toBeGreaterThan(50);
    expect(g.anno.perturbation).toBe("global_position_fixed");
  });

  it("round-trips: global placement seen from the anchor frame equals the relative one", () => {
    // The load-bearing invariant. place_trace_global is place_trace_relative
    // followed by a lift into the world, so bringing it back down into the anchor
    // frame must reproduce the ego-relative placement exactly.
    const anchor = pose(0.7, 100, -50);
    const seed = 17;
    const rel = placeTraceRelative(libStub(), { seed });
    const glob = placeTraceGlobal(libStub(), { seed, anchorPose: anchor });
    const back = transformSpoofToEgoFrame(glob.points, glob.box, anchor);

    expect(back.box.slice(0, 3)).toBeCloseToArray(rel.box.slice(0, 3), 1e-6);
    expect(back.box[6]).toBeCloseTo(rel.box[6], 1e-6);
    expect(back.points.x).toBeCloseToArray(rel.points.x, 1e-6);
    expect(back.points.y).toBeCloseToArray(rel.points.y, 1e-6);
    expect(back.points.z).toBeCloseToArray(rel.points.z, 1e-6);
  });

  it("keeps the phantom fixed in the world as the ego drives past", () => {
    // This is what a world-fixed phantom means: same world position, but its
    // ego-relative range shrinks then grows as the vehicle drives by.
    const anchor = pose(0, 0, 0);
    const glob = placeTraceGlobal(libStub(), { seed: 9, anchorPose: anchor });
    const ranges = [];
    for (const tx of [0, 5, 10, 15, 20, 25]) {
      const ego = transformSpoofToEgoFrame(glob.points, glob.box, pose(0, tx, 0));
      ranges.push(Math.hypot(ego.box[0], ego.box[1]));
    }
    // driving forward toward a phantom ~15-20 m ahead: range decreases
    for (let i = 1; i < 4; i++) expect(ranges[i]).toBeLessThan(ranges[i - 1]);
    // and the world position never changed
    const again = transformSpoofToEgoFrame(glob.points, glob.box, pose(0, 0, 0));
    expect(again.box.slice(0, 2)).toBeCloseToArray(glob.box.slice(0, 2), 1e-9);
  });

  it("adds the anchor heading to the box yaw, and subtracts it coming back", () => {
    const anchor = pose(1.1, 0, 0);
    const glob = placeTraceGlobal(libStub(), { seed: 4, anchorPose: anchor });
    const rel = placeTraceRelative(libStub(), { seed: 4 });
    expect(glob.box[6]).toBeCloseTo(rel.box[6] + poseYaw(anchor), 1e-9);
    const back = transformSpoofToEgoFrame(glob.points, glob.box, anchor);
    expect(back.box[6]).toBeCloseTo(rel.box[6], 1e-9);
  });

  it("honours a chosen trace and location, like the relative placement does", () => {
    const anchor = pose(0, 0, 0);
    const g = placeTraceGlobal(libStub(), { seed: 1, anchorPose: anchor, traceIdx: 2, offsetXY: [30, -2] });
    expect(g.anno.traceIdx).toBe(2);
    // with an identity anchor, world coords equal ego coords
    expect(g.box[0]).toBeCloseTo(30, 1e-9);
    expect(g.box[1]).toBeCloseTo(-2, 1e-9);
  });

  it("requires an anchor pose", () => {
    expect(() => placeTraceGlobal(libStub(), { seed: 1 })).toThrow("anchorPose");
  });
});
