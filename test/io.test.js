import { describe, it, expect } from "./harness.js";
import {
  parseAtlasFrame,
  loadAtlasFrame,
  loadManifest,
  parseRawLidar,
  detectStride,
  toPointFrame,
} from "../src/io/atlasframe.js";
import { configureSensor } from "../src/core/sensor.js";

describe("parseAtlasFrame on real exported data", () => {
  it("loads a real KITTI frame and reports its header", async () => {
    const f = await loadAtlasFrame("../assets/frames/kitti-open-road.atlasframe");
    expect(f.header.dataset).toBe("kitti");
    expect(f.header.sensor).toBe("hdl64e");
    expect(f.header.nPoints).toBeGreaterThan(100000);
    expect(f.x).toHaveLength(f.header.nPoints);
    expect(f.y).toHaveLength(f.header.nPoints);
    expect(f.z).toHaveLength(f.header.nPoints);
    expect(f.intensity).toHaveLength(f.header.nPoints);
  });

  it("dequantizes positions into a plausible driving-scene extent", async () => {
    const f = await loadAtlasFrame("../assets/frames/kitti-open-road.atlasframe");
    let maxAbs = 0;
    let minZ = Infinity;
    for (let i = 0; i < f.x.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(f.x[i]), Math.abs(f.y[i]));
      if (f.z[i] < minZ) minZ = f.z[i];
      expect(Number.isFinite(f.x[i])).toBeTruthy();
    }
    expect(maxAbs).toBeGreaterThan(20);
    expect(maxAbs).toBeLessThan(163);
    // the ground is below a roof-mounted sensor
    expect(minZ).toBeLessThan(-1);
  });

  it("normalises KITTI intensity back to 0-1", async () => {
    const f = await loadAtlasFrame("../assets/frames/kitti-open-road.atlasframe");
    let max = 0;
    for (let i = 0; i < f.intensity.length; i++) max = Math.max(max, f.intensity[i]);
    expect(max).toBeGreaterThan(0.2);
    expect(max).toBeLessThanOrEqual(1.0);
  });

  it("keeps nuScenes intensity on its native 0-255 scale", async () => {
    const f = await loadAtlasFrame("../assets/frames/nuscenes-boston.atlasframe");
    expect(f.header.sensor).toBe("hdl32e");
    let max = 0;
    for (let i = 0; i < f.intensity.length; i++) max = Math.max(max, f.intensity[i]);
    expect(max).toBeGreaterThan(1.0);
    expect(max).toBeLessThanOrEqual(255);
  });

  it("carries ground-truth boxes in the lidar frame", async () => {
    const f = await loadAtlasFrame("../assets/frames/kitti-dense-traffic.atlasframe");
    expect(f.header.boxes.length).toBeGreaterThan(10);
    const b = f.header.boxes[0];
    expect(b.center).toHaveLength(3);
    expect(b.size).toHaveLength(3);
    expect(typeof b.yaw).toBe("number");
    expect(typeof b.label).toBe("string");
  });

  it("rejects a buffer that is not an atlasframe", () => {
    const bad = new Uint8Array(64);
    expect(() => parseAtlasFrame(bad.buffer)).toThrow("magic");
  });

  it("keeps quantization error inside the stated 5 mm quantum", async () => {
    const f = await loadAtlasFrame("../assets/frames/kitti-open-road.atlasframe");
    expect(f.header.posQuantum).toBeCloseTo(0.005, 1e-12);
    // Quantization must be lossless at the index level: dividing back by the
    // quantum has to recover the stored integer. The residual is float32 noise
    // (0.005 is not exactly representable), which stays ~1e-4 of one quantum -
    // four orders of magnitude below the 5 mm step, so no position is ambiguous.
    for (let i = 0; i < 200; i++) {
      const k = f.x[i] / f.header.posQuantum;
      expect(Math.abs(k - Math.round(k))).toBeLessThan(1e-3);
    }
  });
});

describe("loadManifest", () => {
  it("lists the built-in scene library", async () => {
    const m = await loadManifest("../assets/frames/manifest.json");
    expect(m.length).toBeGreaterThan(3);
    for (const e of m) {
      expect(typeof e.file).toBe("string");
      expect(typeof e.label).toBe("string");
      expect(["kitti", "nuscenes"]).toContain(e.dataset);
    }
  });
});

describe("toPointFrame", () => {
  it("indexes a real frame onto the sensor lattice", async () => {
    const f = await loadAtlasFrame("../assets/frames/nuscenes-boston.atlasframe");
    const sensor = configureSensor(f.header.sensor);
    const frame = toPointFrame(f, sensor);
    expect(frame.count).toBe(f.header.nPoints);
    expect(frame.occupiedCells).toBeGreaterThan(1000);

    // Returns per occupied cell lands at ~1.63 rather than the ideal 1.0.
    // This is a dataset property, not a modelling error: released clouds ship
    // rectified float32 XYZ in the ego frame, so the original azimuth lattice
    // cannot be recovered exactly and points scatter across neighbouring
    // columns (the same effect ATLAS_REAL_WORLD_VALIDATION.md reports as clean
    // KITTI failing the azimuth-grid invariant at chance level). Measured
    // directly on this frame: nearest-inclination binning gives 1.63 and the
    // dataset's own ring index gives 1.53, so the residual is azimuth, not beam
    // assignment. What matters is that it stays near 1 rather than the ~9 of a
    // 10-sweep accumulated cloud, which would break per-cell physics entirely.
    expect(frame.count / frame.occupiedCells).toBeLessThan(2.0);
  });

  it("indexes a real 64-beam frame within an interactive time budget", async () => {
    const f = await loadAtlasFrame("../assets/frames/kitti-open-road.atlasframe");
    const sensor = configureSensor(f.header.sensor);
    const t0 = performance.now();
    const frame = toPointFrame(f, sensor);
    const dt = performance.now() - t0;
    expect(frame.count).toBeGreaterThan(100000);
    expect(dt).toBeLessThan(2000);
  });
});

describe("parseRawLidar for user uploads", () => {
  it("parses a KITTI-format float32 x,y,z,intensity buffer", () => {
    const src = new Float32Array([1, 2, 3, 0.5, -4, 5, -6, 0.25]);
    const f = parseRawLidar(src.buffer, { stride: 4 });
    expect(f.header.nPoints).toBe(2);
    expect(f.x[0]).toBeCloseTo(1, 1e-6);
    expect(f.z[1]).toBeCloseTo(-6, 1e-6);
    expect(f.intensity[1]).toBeCloseTo(0.25, 1e-6);
  });

  it("parses a nuScenes-format stride-5 buffer, ignoring the ring column", () => {
    const src = new Float32Array([1, 2, 3, 200, 7, 4, 5, 6, 100, 9]);
    const f = parseRawLidar(src.buffer, { stride: 5 });
    expect(f.header.nPoints).toBe(2);
    expect(f.intensity[0]).toBeCloseTo(200, 1e-6);
    expect(f.x[1]).toBeCloseTo(4, 1e-6);
  });

  it("rejects a buffer that does not divide by the stride", () => {
    const src = new Float32Array(7);
    expect(() => parseRawLidar(src.buffer, { stride: 4 })).toThrow("stride");
  });

  it("infers the stride and says when it is ambiguous", () => {
    expect(detectStride(4 * 4 * 3).stride).toBe(4); // 12 floats: only /4 works
    expect(detectStride(4 * 5 * 3).stride).toBe(5); // 15 floats: only /5 works
    const amb = detectStride(4 * 20); // 20 floats: divisible by both
    expect(amb.ambiguous).toBeTruthy();
    expect(amb.stride).toBe(4);
  });

  it("throws on a buffer that is not a multiple of 4 bytes", () => {
    expect(() => detectStride(13)).toThrow("float32");
  });
});
