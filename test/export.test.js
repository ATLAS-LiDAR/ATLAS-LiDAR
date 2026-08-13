import { describe, it, expect } from "./harness.js";
import { configureSensor } from "../src/core/sensor.js";
import { configureSpoofer } from "../src/core/spoofer.js";
import { frameOf, atCell, emptyFrame } from "./fixtures.js";
import { composeCloud } from "../src/core/compose.js";
import {
  buildAttackBundle,
  parseAttackBundle,
  ATTACK_MAGIC,
  bundleFilename,
} from "../src/io/export.js";

const sensor = configureSensor("hdl64e");

function sampleFrame() {
  return frameOf(sensor, [
    atCell(sensor, 10, 30, 20, 0.4),
    atCell(sensor, 11, 30, 22, 0.5),
    atCell(sensor, 12, 30, 24, 0.6),
  ]);
}

const spoofGt = [30, 1, -0.9, 4.2, 1.8, 1.6, 0.2];

function bundle(opts = {}) {
  const frame = opts.frame ?? sampleFrame();
  const cloud = composeCloud({
    frame,
    keepMask: opts.keepMask ?? null,
    spoof: opts.spoof ?? null,
    replacement: opts.replacement ?? null,
  });
  return buildAttackBundle({
    cloud,
    sensor,
    frameId: "000001",
    dataset: "kitti",
    // `??` would treat an explicit null as absent, which is exactly the case
    // the removal test exercises.
    spoofGt: "spoofGt" in opts ? opts.spoofGt : spoofGt,
    attack: opts.attack ?? { mode: "inject", shape: "box", phantomRange: 30 },
    spooferConfig: configureSpoofer("cao2023"),
    stats: opts.stats ?? { injected: 12, suppressed: 3 },
    scheduledAttack: opts.scheduledAttack ?? true,
  });
}

describe("buildAttackBundle", () => {
  it("writes the magic and a parseable header", () => {
    const b = bundle();
    const parsed = parseAttackBundle(b.buffer);
    expect(parsed.header.magic ?? ATTACK_MAGIC).toBeTruthy();
    expect(parsed.header.frameId).toBe("000001");
    expect(parsed.header.dataset).toBe("kitti");
    expect(parsed.header.sensor).toBe("hdl64e");
  });

  it("rejects a buffer that is not an attack bundle", () => {
    expect(() => parseAttackBundle(new Uint8Array(64).buffer)).toThrow("magic");
  });

  it("stores points as interleaved float32 x,y,z,intensity", () => {
    // KITTI's native layout, so numpy can read the payload with a plain
    // np.frombuffer(..., dtype=float32).reshape(-1, 4).
    const b = bundle();
    const parsed = parseAttackBundle(b.buffer);
    expect(parsed.header.stride).toBe(4);
    expect(parsed.points).toHaveLength(parsed.header.nPoints * 4);
    expect(parsed.points.constructor).toBe(Float32Array);
  });

  it("aligns the payload so a float32 view is valid", () => {
    // Learned the hard way on .atlasframe: a JSON header of arbitrary length
    // pushes the payload to an unaligned offset and typed-array views throw.
    for (const frameId of ["1", "0001", "000001", "a-longer-frame-identifier"]) {
      const cloud = composeCloud({ frame: sampleFrame() });
      const b = buildAttackBundle({
        cloud,
        sensor,
        frameId,
        dataset: "kitti",
        spoofGt,
        attack: {},
        spooferConfig: configureSpoofer("cao2023"),
        stats: {},
      });
      const parsed = parseAttackBundle(b.buffer);
      expect(parsed.header.payloadOffset % 4).toBe(0);
      expect(parsed.points).toHaveLength(parsed.header.nPoints * 4);
    }
  });

  it("round-trips point values", () => {
    const frame = sampleFrame();
    const b = bundle({ frame });
    const parsed = parseAttackBundle(b.buffer);
    expect(parsed.header.nPoints).toBe(frame.count);
    expect(parsed.points[0]).toBeCloseTo(frame.x[0], 1e-3);
    expect(parsed.points[1]).toBeCloseTo(frame.y[0], 1e-3);
    expect(parsed.points[3]).toBeCloseTo(frame.intensity[0], 1e-5);
  });

  it("carries the phantom box as the 7-vector the eval harness reads", () => {
    // eval_matrix_kitti.py does np.asarray(spoof_gt[:7]) and compares 3D IoU.
    const parsed = parseAttackBundle(bundle().buffer);
    expect(parsed.header.spoofGt).toHaveLength(7);
    expect(parsed.header.spoofGt[3]).toBeCloseTo(4.2, 1e-6);
  });

  it("omits spoofGt for a removal attack, since there is no phantom", () => {
    const parsed = parseAttackBundle(bundle({ spoofGt: null }).buffer);
    expect(parsed.header.spoofGt).toBe(null);
  });

  it("reflects destroyed and injected points in the payload", () => {
    const frame = sampleFrame();
    const keepMask = new Uint8Array(frame.count).fill(1);
    keepMask[0] = 0;
    const spoof = {
      x: Float64Array.from([30, 31]),
      y: Float64Array.from([0, 0]),
      z: Float64Array.from([-1, -1]),
      intensity: Float32Array.from([0.5, 0.5]),
    };
    const parsed = parseAttackBundle(bundle({ frame, keepMask, spoof }).buffer);
    expect(parsed.header.nPoints).toBe(frame.count - 1 + 2);
  });

  it("records the attack configuration so a run is reproducible", () => {
    const parsed = parseAttackBundle(
      bundle({ attack: { mode: "remove", attackerRangeM: 0.2, seed: 11 } }).buffer
    );
    expect(parsed.header.attack.mode).toBe("remove");
    expect(parsed.header.attack.seed).toBe(11);
    expect(parsed.header.spoofer.powerRatio).toBeGreaterThan(0);
    expect(parsed.header.spoofer.canPredict).toBeTruthy();
  });

  it("records the scheduled_attack flag the harness distinguishes from dropout", () => {
    // run_injection records both scheduled and realised, so a phantom filtered
    // out by occlusion is never mistaken for a scheduled clean frame.
    expect(parseAttackBundle(bundle({ scheduledAttack: true }).buffer).header.scheduledAttack).toBe(true);
    expect(parseAttackBundle(bundle({ scheduledAttack: false }).buffer).header.scheduledAttack).toBe(false);
  });

  it("stamps the proxy warning so exported data cannot be mistaken for real ASR", () => {
    const parsed = parseAttackBundle(bundle().buffer);
    expect(parsed.header.note.toLowerCase()).toContain("proxy");
  });

  it("handles an empty cloud", () => {
    const cloud = composeCloud({ frame: emptyFrame(sensor) });
    const b = buildAttackBundle({
      cloud,
      sensor,
      frameId: "000002",
      dataset: "kitti",
      spoofGt: null,
      attack: {},
      spooferConfig: configureSpoofer("cao2023"),
      stats: {},
    });
    const parsed = parseAttackBundle(b.buffer);
    expect(parsed.header.nPoints).toBe(0);
    expect(parsed.points).toHaveLength(0);
  });
});

describe("bundleFilename", () => {
  it("encodes frame and attack mode for readability", () => {
    const n = bundleFilename({ frameId: "000001", mode: "inject", slug: "kitti-open-road" });
    expect(n.endsWith(".atlasattack")).toBeTruthy();
    expect(n).toContain("000001");
    expect(n).toContain("inject");
  });

  it("does not produce path separators or spaces", () => {
    const n = bundleFilename({ frameId: "a/b c", mode: "remove", slug: "x y/z" });
    expect(n.includes("/")).toBeFalsy();
    expect(n.includes(" ")).toBeFalsy();
  });
});
