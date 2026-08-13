import { describe, it, expect } from "./harness.js";
import { rng, uniform } from "./util.js";
import { configureSensor } from "../src/core/sensor.js";
import { frameOf, emptyFrame } from "./fixtures.js";
import { composeCloud } from "../src/core/compose.js";
import { detect, CLASS_GROUPS, classifyBySize } from "../src/core/detect.js";
import { iou3d, normalizeBox, maxIoU } from "../src/core/iou.js";
import { loadAtlasFrame, toPointFrame } from "../src/io/atlasframe.js";

const sensor = configureSensor("hdl64e");

/** A dense box-shaped blob of points, like a vehicle's visible surface. */
function blob(cx, cy, cz, dx, dy, dz, n, seed = 1) {
  const next = rng(seed);
  const pts = [];
  for (let i = 0; i < n; i++) {
    pts.push([
      cx + uniform(next, -dx / 2, dx / 2),
      cy + uniform(next, -dy / 2, dy / 2),
      cz + uniform(next, -dz / 2, dz / 2),
      0.3,
    ]);
  }
  return pts;
}

/** A flat ground plane patch. */
function ground(x0, x1, y0, y1, z, step = 0.5) {
  const pts = [];
  for (let x = x0; x <= x1; x += step) {
    for (let y = y0; y <= y1; y += step) pts.push([x, y, z, 0.15]);
  }
  return pts;
}

describe("composeCloud", () => {
  it("returns the frame unchanged with no perturbation", () => {
    const frame = frameOf(sensor, blob(20, 0, 0, 2, 2, 1, 50));
    const c = composeCloud({ frame });
    expect(c.n).toBe(frame.count);
    expect(c.x[0]).toBeCloseTo(frame.x[0], 1e-6);
  });

  it("drops points the keepMask zeroes", () => {
    const frame = frameOf(sensor, blob(20, 0, 0, 2, 2, 1, 50));
    const keepMask = new Uint8Array(frame.count).fill(1);
    for (let i = 0; i < 10; i++) keepMask[i] = 0;
    const c = composeCloud({ frame, keepMask });
    expect(c.n).toBe(frame.count - 10);
  });

  it("appends spoofed points and tags their origin", () => {
    const frame = frameOf(sensor, blob(20, 0, 0, 2, 2, 1, 50));
    const spoof = {
      x: Float64Array.from([5, 6]),
      y: Float64Array.from([0, 0]),
      z: Float64Array.from([0, 0]),
      intensity: Float32Array.from([1, 1]),
    };
    const c = composeCloud({ frame, spoof });
    expect(c.n).toBe(frame.count + 2);
    expect(c.origin[c.n - 1]).toBe(1);
    expect(c.origin[0]).toBe(0);
  });

  it("appends range-replaced points with their own origin tag", () => {
    const frame = frameOf(sensor, blob(20, 0, 0, 2, 2, 1, 20));
    const replacement = {
      x: Float64Array.from([3]),
      y: Float64Array.from([0]),
      z: Float64Array.from([0]),
      intensity: Float32Array.from([255]),
    };
    const c = composeCloud({ frame, replacement });
    expect(c.origin[c.n - 1]).toBe(2);
  });
});

describe("classifyBySize", () => {
  it("uses the ATLAS class groups", () => {
    expect(CLASS_GROUPS).toContain("vehicle");
    expect(CLASS_GROUPS).toContain("pedestrian");
    expect(CLASS_GROUPS).toContain("cyclist");
  });

  it("calls a car-sized footprint a vehicle", () => {
    expect(classifyBySize([4.2, 1.8, 1.6])).toBe("vehicle");
    expect(classifyBySize([12, 2.6, 3.2])).toBe("vehicle");
  });

  it("calls a narrow tall footprint a pedestrian", () => {
    expect(classifyBySize([0.7, 0.6, 1.75])).toBe("pedestrian");
  });

  it("calls an intermediate elongated-but-narrow footprint a cyclist", () => {
    expect(classifyBySize([1.7, 0.6, 1.7])).toBe("cyclist");
  });
});

describe("detect on synthetic scenes", () => {
  it("finds nothing in an empty frame", () => {
    const dets = detect(composeCloud({ frame: emptyFrame(sensor) }), { sensor });
    expect(dets).toHaveLength(0);
  });

  it("finds nothing in bare ground, so ground removal works", () => {
    // A clustering detector that keeps the ground finds one enormous blob and is
    // useless; this is the load-bearing precondition.
    const frame = frameOf(sensor, ground(5, 40, -10, 10, -1.7));
    const dets = detect(composeCloud({ frame }), { sensor });
    expect(dets).toHaveLength(0);
  });

  it("finds a vehicle-sized blob standing on the ground", () => {
    const frame = frameOf(sensor, [
      ...ground(5, 40, -10, 10, -1.7),
      ...blob(20, 0, -0.9, 4.2, 1.8, 1.6, 400),
    ]);
    const dets = detect(composeCloud({ frame }), { sensor });
    expect(dets.length).toBeGreaterThan(0);
    const best = dets.reduce((a, b) => (a.nPoints > b.nPoints ? a : b));
    expect(best.center[0]).toBeCloseTo(20, 1.5);
    expect(best.center[1]).toBeCloseTo(0, 1.5);
    expect(best.label).toBe("vehicle");
  });

  it("separates two objects that are far apart", () => {
    const frame = frameOf(sensor, [
      ...ground(5, 60, -14, 14, -1.7),
      ...blob(20, -6, -0.9, 4.2, 1.8, 1.6, 400, 2),
      ...blob(40, 8, -0.9, 4.2, 1.8, 1.6, 400, 3),
    ]);
    const dets = detect(composeCloud({ frame }), { sensor });
    const big = dets.filter((d) => d.nPoints > 100);
    expect(big.length).toBeGreaterThanOrEqual(2);
  });

  it("recovers a box whose IoU against the true extent is high", () => {
    const truth = normalizeBox({ center: [20, 0, -0.9], size: [4.2, 1.8, 1.6], yaw: 0 });
    const frame = frameOf(sensor, [
      ...ground(5, 40, -10, 10, -1.7),
      ...blob(20, 0, -0.9, 4.2, 1.8, 1.6, 600),
    ]);
    const dets = detect(composeCloud({ frame }), { sensor });
    const { iou } = maxIoU(dets, truth);
    expect(iou).toBeGreaterThan(0.3);
  });

  it("scores a denser cluster higher than a sparse one", () => {
    const dense = detect(
      composeCloud({
        frame: frameOf(sensor, [...ground(5, 40, -10, 10, -1.7), ...blob(20, 0, -0.9, 4.2, 1.8, 1.6, 600, 5)]),
      }),
      { sensor }
    );
    const sparse = detect(
      composeCloud({
        frame: frameOf(sensor, [...ground(5, 40, -10, 10, -1.7), ...blob(20, 0, -0.9, 4.2, 1.8, 1.6, 40, 5)]),
      }),
      { sensor }
    );
    expect(dense.length).toBeGreaterThan(0);
    expect(sparse.length).toBeGreaterThan(0);
    const s = (d) => Math.max(...d.map((x) => x.score));
    expect(s(dense)).toBeGreaterThan(s(sparse));
  });

  it("gives every detection a score in [0, 1]", () => {
    const frame = frameOf(sensor, [...ground(5, 40, -10, 10, -1.7), ...blob(20, 0, -0.9, 4.2, 1.8, 1.6, 400)]);
    for (const d of detect(composeCloud({ frame }), { sensor })) {
      expect(d.score).toBeGreaterThanOrEqual(0);
      expect(d.score).toBeLessThanOrEqual(1);
    }
  });

  it("is deterministic", () => {
    const frame = frameOf(sensor, [...ground(5, 40, -10, 10, -1.7), ...blob(20, 0, -0.9, 4.2, 1.8, 1.6, 400)]);
    const a = detect(composeCloud({ frame }), { sensor });
    const b = detect(composeCloud({ frame }), { sensor });
    expect(a.length).toBe(b.length);
    expect(a[0].center).toBeCloseToArray(b[0].center, 1e-12);
  });
});

describe("detect on a real recorded frame", () => {
  // The credibility test for the whole proxy. If it cannot recover real GT
  // vehicles from a real cloud, every ASR number built on it is noise.
  it("recovers a useful fraction of nearby ground-truth vehicles", async () => {
    const loaded = await loadAtlasFrame("../assets/frames/kitti-dense-traffic.atlasframe");
    const sn = configureSensor(loaded.header.sensor);
    const frame = toPointFrame(loaded, sn);
    const dets = detect(composeCloud({ frame }), { sensor: sn });

    expect(dets.length).toBeGreaterThan(2);

    const nearbyVehicles = loaded.header.boxes.filter((b) => {
      const r = Math.hypot(b.center[0], b.center[1]);
      return r < 40 && /Car|Van|Truck/i.test(b.label);
    });
    expect(nearbyVehicles.length).toBeGreaterThan(0);

    let recovered = 0;
    for (const gt of nearbyVehicles) {
      if (maxIoU(dets, normalizeBox(gt)).iou >= 0.3) recovered++;
    }
    const rate = recovered / nearbyVehicles.length;
    // A clustering proxy is not a trained detector; require only that it is
    // clearly better than chance, and record the measured rate in the message.
    expect(rate).toBeGreaterThan(0.25);
  });

  it("runs on a full 120k-point frame inside an interactive budget", async () => {
    const loaded = await loadAtlasFrame("../assets/frames/kitti-open-road.atlasframe");
    const sn = configureSensor(loaded.header.sensor);
    const frame = toPointFrame(loaded, sn);
    const cloud = composeCloud({ frame });
    const t0 = performance.now();
    detect(cloud, { sensor: sn });
    expect(performance.now() - t0).toBeLessThan(2500);
  });
});
