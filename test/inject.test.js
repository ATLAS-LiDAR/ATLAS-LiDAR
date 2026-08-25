import { describe, it, expect } from "./harness.js";
import { configureSensor } from "../src/core/sensor.js";
import { configureSpoofer, reachableCells } from "../src/core/spoofer.js";
import { frameOf, emptyFrame, atCell } from "./fixtures.js";
import { sphericalToCart, cellKey } from "../src/core/polar.js";
import { injectPhantom, groundAnchorZ, percentile5 } from "../src/core/inject.js";

const sensor = configureSensor("hdl32e");
const CENTRE_ROW = 23; // beam nearest 0 deg elevation on the HDL-32E
const SPOOF_POS = [40, 0, 0]; // bearing 0, standoff 40 m

/** Phantom points as {x,y,z} arrays, built on exact lattice cells. */
function phantomAt(cells) {
  const x = [];
  const y = [];
  const z = [];
  for (const [azIdx, row, r] of cells) {
    const az = (azIdx + 0.5) * sensor.azRes;
    const el = sensor.inclinations[row];
    const [px, py, pz] = sphericalToCart(r, az, el);
    x.push(px);
    y.push(py);
    z.push(pz);
  }
  return { x: Float64Array.from(x), y: Float64Array.from(y), z: Float64Array.from(z) };
}

function inject(opts) {
  return injectPhantom({
    sensor,
    spoofer: configureSpoofer("cao2023"),
    positionXYZ: SPOOF_POS,
    dataset: "nuscenes",
    intensityMode: "range-conditioned",
    seed: 7,
    groundAnchor: false,
    ...opts,
  });
}

describe("injectPhantom occlusion physics", () => {
  it("injects a phantom that is the first return on its ray", () => {
    const res = inject({
      frame: emptyFrame(sensor),
      phantom: phantomAt([[0, CENTRE_ROW, 60]]),
    });
    expect(res.stats.injected).toBe(1);
    expect(res.spoof.r).toHaveLength(1);
    expect(res.spoof.r[0]).toBeCloseTo(60, 0.2);
  });

  it("suppresses a real return that lies behind the phantom", () => {
    // Single-return sensor: the victim reports only the nearest echo, so an
    // injected point in front destroys the genuine one behind it.
    const frame = frameOf(sensor, [atCell(sensor, 0, CENTRE_ROW, 80)]);
    const res = inject({ frame, phantom: phantomAt([[0, CENTRE_ROW, 60]]) });
    expect(res.stats.injected).toBe(1);
    expect(res.stats.suppressed).toBe(1);
    expect(res.keepMask[0]).toBe(0);
  });

  it("drops a phantom occluded by real geometry in front of it", () => {
    const frame = frameOf(sensor, [atCell(sensor, 0, CENTRE_ROW, 50)]);
    const res = inject({ frame, phantom: phantomAt([[0, CENTRE_ROW, 60]]) });
    expect(res.stats.injected).toBe(0);
    expect(res.stats.rejected.occluded).toBe(1);
    expect(res.keepMask[0]).toBe(1);
  });

  it("keeps a real return whose range EXACTLY ties the phantom's", () => {
    // ATLAS has a third branch for r == r_group[0]:
    //   k = searchsorted(r_group, r, side="right"); remove idx_group[k:]
    // Because the search is right-sided it steps PAST the tied returns, so the
    // tied real return survives and only strictly-farther ones die. Collapsing
    // this into the "phantom in front" branch kills one extra point - which is
    // exactly the off-by-one tools/crossvalidate.py caught.
    const frame = frameOf(sensor, [atCell(sensor, 0, CENTRE_ROW, 60)]);
    // read the coordinates back OUT of the frame, so the phantom's range is
    // computed from bit-identical inputs and the tie is exact
    const phantom = {
      x: Float64Array.from([frame.x[0]]),
      y: Float64Array.from([frame.y[0]]),
      z: Float64Array.from([frame.z[0]]),
    };
    const res = inject({ frame, phantom });
    expect(res.stats.injected).toBe(1);
    expect(res.stats.suppressed).toBe(0);
    expect(res.keepMask[0]).toBe(1);
  });

  it("still removes returns strictly beyond a tied phantom", () => {
    const frame = frameOf(sensor, [
      atCell(sensor, 0, CENTRE_ROW, 60),
      atCell(sensor, 0, CENTRE_ROW, 80),
    ]);
    const near = frame.order[frame.cellRange(frame.cell[0]).start];
    const phantom = {
      x: Float64Array.from([frame.x[near]]),
      y: Float64Array.from([frame.y[near]]),
      z: Float64Array.from([frame.z[near]]),
    };
    const res = inject({ frame, phantom });
    expect(res.stats.injected).toBe(1);
    expect(res.stats.suppressed).toBe(1); // the 80 m return, not the tied one
  });

  it("keeps the nearest real return alongside the phantom on a dual-return sensor", () => {
    const dual = configureSensor("hdl32e", { returnMode: "dual" });
    const frame = frameOf(dual, [atCell(dual, 0, CENTRE_ROW, 80)]);
    const res = injectPhantom({
      frame,
      sensor: dual,
      spoofer: configureSpoofer("cao2023"),
      positionXYZ: SPOOF_POS,
      dataset: "nuscenes",
      intensityMode: "range-conditioned",
      seed: 3,
      groundAnchor: false,
      phantom: phantomAt([[0, CENTRE_ROW, 60]]),
    });
    expect(res.stats.injected).toBe(1);
    // pTwo ~ 0.98, so the genuine echo usually survives as the second return
    expect(res.stats.suppressed).toBe(0);
    expect(res.keepMask[0]).toBe(1);
  });
});

describe("injectPhantom physical gates", () => {
  it("rejects a phantom outside the spoofer's reachable envelope", () => {
    // 90 degrees off the spoofer bearing: no power reaches those cells.
    const far = Math.round(Math.PI / 2 / sensor.azRes);
    const res = inject({
      frame: emptyFrame(sensor),
      phantom: phantomAt([[far, CENTRE_ROW, 60]]),
    });
    expect(res.stats.injected).toBe(0);
    expect(res.stats.rejected.unreachable).toBe(1);
  });

  it("rejects a phantom nearer than the relay floor when the spoofer cannot predict", () => {
    // The physics the MVP could not express: a reactive spoofer 40 m away cannot
    // put a phantom at 10 m without firing before the victim's pulse arrives.
    const reactive = configureSpoofer("cao2023", { canPredict: false });
    const res = inject({
      frame: emptyFrame(sensor),
      spoofer: reactive,
      phantom: phantomAt([[0, CENTRE_ROW, 10]]),
    });
    expect(res.stats.injected).toBe(0);
    expect(res.stats.rejected.tof).toBe(1);
  });

  it("allows the same near-field phantom once pulse prediction is enabled", () => {
    const predictive = configureSpoofer("cao2023", { canPredict: true });
    const res = inject({
      frame: emptyFrame(sensor),
      spoofer: predictive,
      phantom: phantomAt([[0, CENTRE_ROW, 10]]),
    });
    expect(res.stats.injected).toBe(1);
    expect(res.stats.rejected.tof).toBe(0);
  });

  it("rejects a phantom below the sensor's minimum operational threshold", () => {
    const predictive = configureSpoofer("cao2023", { canPredict: true });
    const res = inject({
      frame: emptyFrame(sensor),
      spoofer: predictive,
      phantom: phantomAt([[0, CENTRE_ROW, 0.2]]),
    });
    expect(res.stats.injected).toBe(0);
    expect(res.stats.rejected.mot).toBe(1);
  });

  it("rejects a phantom beyond the victim's range gate", () => {
    const res = inject({
      frame: emptyFrame(sensor),
      phantom: phantomAt([[0, CENTRE_ROW, 400]]),
    });
    expect(res.stats.injected).toBe(0);
    expect(res.stats.rejected.tof).toBe(1);
  });

  it("enforces one spoofed return per cell, counting cells not candidates", () => {
    // The MVP counted every aim target as an injected point even when many
    // collapsed onto the same beam, so the readout disagreed with the picture.
    const res = inject({
      frame: emptyFrame(sensor),
      phantom: phantomAt([
        [0, CENTRE_ROW, 60],
        [0, CENTRE_ROW, 62],
        [0, CENTRE_ROW, 64],
      ]),
    });
    expect(res.stats.requested).toBe(3);
    expect(res.stats.injected).toBe(1);
    expect(res.stats.rejected.cellConflict).toBe(2);
    expect(res.spoof.r).toHaveLength(1);
    // the nearest candidate wins, as a first/strongest-return sensor would report
    expect(res.spoof.r[0]).toBeCloseTo(60, 0.2);
  });

  it("never fills more cells than the spoofer can reach", () => {
    const cells = [];
    for (let a = -40; a <= 40; a++) {
      for (let row = 0; row < sensor.nBeams; row++) {
        cells.push([(a + sensor.azCellCount) % sensor.azCellCount, row, 60]);
      }
    }
    const spoofer = configureSpoofer("cao2023");
    const reach = reachableCells({
      frame: emptyFrame(sensor),
      sensor,
      spoofer,
      positionXYZ: SPOOF_POS,
    });
    const res = inject({ frame: emptyFrame(sensor), phantom: phantomAt(cells), spoofer });
    expect(res.stats.injected).toBeLessThanOrEqual(reach.cells.length);
    expect(res.stats.fill).toBeLessThanOrEqual(1.0);
  });
});

describe("injectPhantom synchronization", () => {
  function spreadPhantom() {
    const cells = [];
    for (let a = -8; a <= 8; a++) {
      for (let row = CENTRE_ROW - 6; row <= CENTRE_ROW + 6; row++) {
        cells.push([(a + sensor.azCellCount) % sensor.azCellCount, row, 60]);
      }
    }
    return phantomAt(cells);
  }

  it("lands far fewer points when the spoofer is free-running", () => {
    // An unsynchronized spoofer only hits the receiver's open gate ~2% of the
    // time. This is why every practical attack locks to the victim's PRF.
    const phantom = spreadPhantom();
    const locked = inject({
      frame: emptyFrame(sensor),
      phantom,
      spoofer: configureSpoofer("cao2023", { syncMode: "locked", jitterS: 1e-12 }),
    });
    const free = inject({
      frame: emptyFrame(sensor),
      phantom,
      spoofer: configureSpoofer("cao2023", { syncMode: "free" }),
    });
    expect(locked.stats.injected).toBeGreaterThan(10);
    expect(free.stats.injected * 5).toBeLessThan(locked.stats.injected);
    // and the loss must be attributed to sync, not to angular response
    expect(free.stats.rejected.sync).toBeGreaterThan(0);
    expect(free.stats.rejected.sync).toBeGreaterThan(free.stats.rejected.response);
  });

  it("degrades with lock jitter", () => {
    const phantom = spreadPhantom();
    const tight = inject({
      frame: emptyFrame(sensor),
      phantom,
      spoofer: configureSpoofer("cao2023", { jitterS: 1e-12 }),
    });
    const loose = inject({
      frame: emptyFrame(sensor),
      phantom,
      spoofer: configureSpoofer("cao2023", { jitterS: 30e-9 }),
    });
    expect(loose.stats.injected).toBeLessThan(tight.stats.injected);
  });

  it("is reproducible for a fixed seed and varies with it", () => {
    const phantom = spreadPhantom();
    const cfg = { frame: emptyFrame(sensor), phantom, spoofer: configureSpoofer("cao2023", { jitterS: 20e-9 }) };
    const a = inject({ ...cfg, seed: 11 });
    const b = inject({ ...cfg, seed: 11 });
    const c = inject({ ...cfg, seed: 12 });
    expect(a.stats.injected).toBe(b.stats.injected);
    expect(a.stats.injected === c.stats.injected && a.stats.rejected.sync === c.stats.rejected.sync).toBeFalsy();
  });
});

describe("groundAnchorZ", () => {
  it("drops the phantom so its base sits on the local ground", () => {
    // ATLAS anchors to the 5th percentile of scene z within a 2 m radius, so a
    // phantom car does not float or sink into the road.
    // A dense ground patch: ATLAS requires >20 support points inside the radius
    // before it will anchor, so a sparse arc is not enough.
    const ground = [];
    for (let i = -7; i <= 7; i++) {
      for (let j = -7; j <= 7; j++) {
        ground.push([30 + i * 0.25, j * 0.25, -1.8, 0.2]);
      }
    }
    const frame = frameOf(sensor, ground);
    const phantom = { x: Float64Array.from([30, 30.4]), y: Float64Array.from([0.2, -0.2]), z: Float64Array.from([3.0, 4.5]) };
    const dz = groundAnchorZ({ frame, phantom, radiusM: 2.0 });
    expect(dz).toBeCloseTo(-1.8 - 3.0, 0.2);
  });

  it("returns zero when there is no local ground to anchor to", () => {
    const dz = groundAnchorZ({
      frame: emptyFrame(sensor),
      phantom: { x: Float64Array.from([30]), y: Float64Array.from([0]), z: Float64Array.from([1]) },
      radiusM: 2.0,
    });
    expect(dz).toBe(0);
  });

  // ATLAS calls np.percentile(z, 5), whose default method LINEARLY INTERPOLATES
  // between the two order statistics straddling position 0.05*(n-1). Taking the
  // lower neighbour instead shifts dz by a fraction of the gap between adjacent
  // z values - only millimetres, but enough to push a phantom point across a
  // beam boundary and destroy a different scene point. Reference values come
  // from numpy itself.
  it("matches numpy's linearly-interpolated 5th percentile", () => {
    expect(percentile5(Float64Array.from({ length: 22 }, (_, i) => i))).toBeCloseTo(1.05, 1e-12);
    expect(percentile5(Float64Array.from({ length: 21 }, (_, i) => i))).toBeCloseTo(1.0, 1e-12);
    expect(percentile5(Float64Array.from([0, 3, 7, ...Array(19).fill(10)]))).toBeCloseTo(3.2, 1e-12);
    expect(percentile5(Float64Array.from([1, 5]))).toBeCloseTo(1.2, 1e-12);
    expect(percentile5(Float64Array.from([4]))).toBeCloseTo(4, 1e-12);
  });

  it("interpolates the ground percentile rather than snapping to a sample", () => {
    // 22 support points whose 5th percentile falls BETWEEN two samples: 0.05*21
    // = 1.05, so the answer is 5% of the way from the 2nd to the 3rd value.
    const ground = [];
    const zs = [-1.9, -1.85, -1.7];
    for (let i = 0; i < 22; i++) {
      ground.push([30 + (i % 5) * 0.2, (i % 7) * 0.2 - 0.6, zs[Math.min(i, 2)], 0.2]);
    }
    const frame = frameOf(sensor, ground);
    const phantom = { x: Float64Array.from([30]), y: Float64Array.from([0]), z: Float64Array.from([0]) };
    const dz = groundAnchorZ({ frame, phantom, radiusM: 2.0 });
    // sorted z = [-1.9, -1.85, -1.7, -1.7, ...] -> p5 = -1.85 + 0.05*(-1.7 + 1.85)
    // the fixture's cloud arrives as a float32 buffer, so the tolerance sits above
    // float32 noise (~1e-7 at this magnitude) rather than at float64 exactness
    expect(dz).toBeCloseTo(-1.85 + 0.05 * 0.15, 1e-6);
  });

  it("is applied by injectPhantom when enabled", () => {
    const ground = [];
    for (let i = -7; i <= 7; i++) {
      for (let j = -7; j <= 7; j++) {
        ground.push([60 + i * 0.25, j * 0.25, -1.9, 0.2]);
      }
    }
    const frame = frameOf(sensor, ground);
    const phantom = phantomAt([[0, CENTRE_ROW, 60]]);
    const anchored = inject({ frame, phantom, groundAnchor: true });
    const raw = inject({ frame, phantom, groundAnchor: false });
    expect(anchored.stats.groundShiftZ).not.toBe(0);
    expect(raw.stats.groundShiftZ).toBe(0);
  });
});

describe("injectPhantom output", () => {
  it("assigns range-conditioned intensity to spoofed points", () => {
    const res = inject({
      frame: emptyFrame(sensor),
      phantom: phantomAt([[0, CENTRE_ROW, 25]]),
    });
    expect(res.spoof.intensity[0]).toBeGreaterThan(0);
    expect(res.spoof.intensity[0]).toBeLessThanOrEqual(255);
  });

  it("assigns saturated intensity when asked, modelling the real spoofer", () => {
    const res = inject({
      frame: emptyFrame(sensor),
      phantom: phantomAt([[0, CENTRE_ROW, 25]]),
      intensityMode: "saturated",
    });
    expect(res.spoof.intensity[0]).toBe(255);
  });

  it("returns a keepMask covering every point of the original frame", () => {
    const frame = frameOf(sensor, [
      atCell(sensor, 0, CENTRE_ROW, 80),
      atCell(sensor, 500, 10, 30),
    ]);
    const res = inject({ frame, phantom: phantomAt([[0, CENTRE_ROW, 60]]) });
    expect(res.keepMask).toHaveLength(frame.count);
    expect(res.keepMask[1]).toBe(1); // untouched cell survives
  });

  it("reports every rejection reason so nothing vanishes unexplained", () => {
    const res = inject({
      frame: emptyFrame(sensor),
      phantom: phantomAt([[0, CENTRE_ROW, 60]]),
    });
    const r = res.stats.rejected;
    for (const k of ["unreachable", "response", "tof", "mot", "sync", "cellConflict", "occluded"]) {
      expect(typeof r[k]).toBe("number");
    }
    const total =
      res.stats.injected +
      r.unreachable + r.response + r.tof + r.mot + r.sync + r.cellConflict + r.occluded;
    expect(total).toBe(res.stats.requested);
  });
});

describe("injectPhantom in ATLAS-faithful mode", () => {
  // ATLAS's spoof_frame applies NO reachability, timing or power model, and it
  // evaluates each phantom point independently with no per-cell dedupe. Faithful
  // mode must reproduce that, including the consequences.
  const atlas = (opts) => inject({ atlasFaithful: true, ...opts });

  it("injects outside the spoofer envelope, because ATLAS has no envelope", () => {
    const far = Math.round(Math.PI / 2 / sensor.azRes);
    const res = atlas({
      frame: emptyFrame(sensor),
      phantom: phantomAt([[far, CENTRE_ROW, 60]]),
    });
    expect(res.stats.injected).toBe(1);
    expect(res.stats.rejected.unreachable).toBe(0);
  });

  it("injects inside the relay floor, because ATLAS has no timing model", () => {
    const res = atlas({
      frame: emptyFrame(sensor),
      spoofer: configureSpoofer("cao2023", { canPredict: false }),
      phantom: phantomAt([[0, CENTRE_ROW, 10]]),
    });
    expect(res.stats.injected).toBe(1);
    expect(res.stats.rejected.tof).toBe(0);
  });

  it("keeps every phantom point in a shared cell, with no dedupe", () => {
    const res = atlas({
      frame: emptyFrame(sensor),
      phantom: phantomAt([
        [0, CENTRE_ROW, 60],
        [0, CENTRE_ROW, 62],
        [0, CENTRE_ROW, 64],
      ]),
    });
    expect(res.stats.injected).toBe(3);
    expect(res.stats.rejected.cellConflict).toBe(0);
    // and the consequence: more than one return per cell, which is what
    // realizability.py is for
    expect(res.stats.fill).toBeGreaterThan(1.0);
  });

  it("still resolves occlusion exactly as spoof_frame does", () => {
    const frame = frameOf(sensor, [atCell(sensor, 0, CENTRE_ROW, 50)]);
    const res = atlas({ frame, phantom: phantomAt([[0, CENTRE_ROW, 60]]) });
    expect(res.stats.injected).toBe(0);
    expect(res.stats.rejected.occluded).toBe(1);
    expect(res.keepMask[0]).toBe(1);
  });

  it("still suppresses real returns behind a phantom that lands in front", () => {
    const frame = frameOf(sensor, [atCell(sensor, 0, CENTRE_ROW, 80)]);
    const res = atlas({ frame, phantom: phantomAt([[0, CENTRE_ROW, 60]]) });
    expect(res.stats.injected).toBe(1);
    expect(res.stats.suppressed).toBe(1);
  });

  it("keeps a phantom in a cell the victim never sampled", () => {
    // key not in key_to_slice -> `continue`, i.e. the phantom survives
    const res = atlas({
      frame: frameOf(sensor, [atCell(sensor, 500, 10, 30)]),
      phantom: phantomAt([[0, CENTRE_ROW, 60]]),
    });
    expect(res.stats.injected).toBe(1);
  });

  it("differs from physics mode on the same input", () => {
    const phantom = phantomAt([[0, CENTRE_ROW, 60], [0, CENTRE_ROW, 62]]);
    const physics = inject({ frame: emptyFrame(sensor), phantom });
    const faithful = atlas({ frame: emptyFrame(sensor), phantom });
    expect(faithful.stats.injected).toBeGreaterThan(physics.stats.injected);
  });

  it("still reconciles its accounting", () => {
    const res = atlas({
      frame: emptyFrame(sensor),
      phantom: phantomAt([[0, CENTRE_ROW, 60], [0, CENTRE_ROW, 62]]),
    });
    const r = res.stats.rejected;
    const total = res.stats.injected + r.unreachable + r.response + r.tof + r.mot +
      r.sync + r.cellConflict + r.occluded;
    expect(total).toBe(res.stats.requested);
  });
});
