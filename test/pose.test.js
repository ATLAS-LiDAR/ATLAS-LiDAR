import { describe, it, expect } from "./harness.js";
import { rng, uniform } from "./util.js";
import {
  IDENTITY4,
  matMul4,
  invertRigid4,
  applyPose,
  poseYaw,
  transformPoints,
  relativePose,
} from "../src/core/pose.js";
import { loadSequence, loadSequenceManifest, frameOfSequence } from "../src/io/sequence.js";

/** Rigid transform: rotation about z by `yaw` then translation. */
function rigid(yaw, tx, ty, tz) {
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [c, -s, 0, tx, s, c, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1];
}

describe("pose maths", () => {
  it("applies rotation then translation, row-major 4x4", () => {
    const m = rigid(Math.PI / 2, 10, 0, 0);
    expect(applyPose(m, [1, 0, 0])).toBeCloseToArray([10, 1, 0], 1e-9);
    expect(applyPose(IDENTITY4, [3, -4, 5])).toBeCloseToArray([3, -4, 5], 1e-12);
  });

  it("inverts a rigid transform exactly", () => {
    const next = rng(3);
    for (let i = 0; i < 100; i++) {
      const m = rigid(uniform(next, -Math.PI, Math.PI), uniform(next, -50, 50), uniform(next, -50, 50), uniform(next, -3, 3));
      const inv = invertRigid4(m);
      const p = [uniform(next, -30, 30), uniform(next, -30, 30), uniform(next, -5, 5)];
      expect(applyPose(inv, applyPose(m, p))).toBeCloseToArray(p, 1e-9);
      expect(applyPose(m, applyPose(inv, p))).toBeCloseToArray(p, 1e-9);
    }
  });

  it("composes transforms in the expected order", () => {
    const a = rigid(Math.PI / 2, 0, 0, 0);
    const b = rigid(0, 5, 0, 0);
    // matMul4(a, b) should mean "apply b, then a"
    const p = [1, 0, 0];
    expect(applyPose(matMul4(a, b), p)).toBeCloseToArray(applyPose(a, applyPose(b, p)), 1e-9);
  });

  it("reads the yaw ATLAS reads: atan2(m[1][0], m[0][0])", () => {
    for (const yaw of [0, 0.4, -1.2, Math.PI / 2, 3.0]) {
      expect(poseYaw(rigid(yaw, 7, -2, 1))).toBeCloseTo(yaw, 1e-9);
    }
  });

  it("transforms a whole point set", () => {
    const m = rigid(Math.PI / 2, 1, 2, 3);
    const pts = { x: Float64Array.from([1, 0]), y: Float64Array.from([0, 1]), z: Float64Array.from([0, 0]) };
    const out = transformPoints(m, pts);
    expect([out.x[0], out.y[0], out.z[0]]).toBeCloseToArray([1, 3, 3], 1e-9);
    expect([out.x[1], out.y[1], out.z[1]]).toBeCloseToArray([0, 2, 3], 1e-9);
  });

  it("builds the pose that carries frame A's coords into frame B", () => {
    const next = rng(11);
    for (let i = 0; i < 50; i++) {
      const A = rigid(uniform(next, -3, 3), uniform(next, -40, 40), uniform(next, -40, 40), 0);
      const B = rigid(uniform(next, -3, 3), uniform(next, -40, 40), uniform(next, -40, 40), 0);
      const AtoB = relativePose(A, B);
      const p = [uniform(next, -20, 20), uniform(next, -20, 20), 0];
      // going A -> global -> B must equal the direct relative transform
      const viaGlobal = applyPose(invertRigid4(B), applyPose(A, p));
      expect(applyPose(AtoB, p)).toBeCloseToArray(viaGlobal, 1e-8);
    }
  });
});

describe("sequence loading", () => {
  it("lists the exported sequences", async () => {
    const m = await loadSequenceManifest("../assets/sequences/manifest.json");
    expect(m.length).toBeGreaterThan(0);
    for (const e of m) {
      expect(e.dataset).toBe("nuscenes");
      expect(e.nFrames).toBeGreaterThan(1);
      expect(e.dtSec).toBeCloseTo(0.5, 0.05);
    }
  });

  it("loads a real sequence with per-frame poses and boxes", async () => {
    const seq = await loadSequence("../assets/sequences/nuscenes-scene0.atlasseq");
    expect(seq.header.nFrames).toBe(20);
    expect(seq.header.sensor).toBe("hdl32e");
    // nuScenes keyframes are 2 Hz; match_with_velocity needs this, not 0.1
    expect(seq.header.dtSec).toBeCloseTo(0.5, 0.05);
    expect(seq.header.frames).toHaveLength(20);
    for (const f of seq.header.frames) {
      expect(f.pose).toHaveLength(16);
      expect(f.nPoints).toBeGreaterThan(1000);
    }
  });

  it("slices out an individual frame's cloud", async () => {
    const seq = await loadSequence("../assets/sequences/nuscenes-scene0.atlasseq");
    for (const k of [0, 9, 19]) {
      const f = frameOfSequence(seq, k);
      expect(f.x).toHaveLength(seq.header.counts[k]);
      expect(f.intensity.length).toBe(f.x.length);
      expect(f.header.sensor).toBe("hdl32e");
      expect(f.header.boxes.length).toBeGreaterThan(0);
      let finite = true;
      for (let i = 0; i < f.x.length; i++) if (!Number.isFinite(f.x[i])) finite = false;
      expect(finite).toBeTruthy();
    }
  });

  it("rejects an out-of-range frame index", async () => {
    const seq = await loadSequence("../assets/sequences/nuscenes-scene0.atlasseq");
    expect(() => frameOfSequence(seq, 99)).toThrow("range");
  });

  it("has orthonormal rotations in every pose", async () => {
    const seq = await loadSequence("../assets/sequences/nuscenes-scene0.atlasseq");
    for (const f of seq.header.frames) {
      const m = f.pose;
      // columns of R must be unit length
      for (let c = 0; c < 3; c++) {
        const n = Math.hypot(m[c], m[4 + c], m[8 + c]);
        expect(n).toBeCloseTo(1, 1e-5);
      }
    }
  });

  it("shows the ego actually moving through the world", async () => {
    const seq = await loadSequence("../assets/sequences/nuscenes-scene0.atlasseq");
    const t = seq.header.frames.map((f) => [f.pose[3], f.pose[7], f.pose[11]]);
    let dist = 0;
    for (let i = 1; i < t.length; i++) {
      dist += Math.hypot(t[i][0] - t[i - 1][0], t[i][1] - t[i - 1][1]);
    }
    expect(dist).toBeGreaterThan(10);
    const span = (seq.header.nFrames - 1) * seq.header.dtSec;
    const kmh = (3.6 * dist) / span;
    expect(kmh).toBeGreaterThan(1);
    expect(kmh).toBeLessThan(130);
  });

  it("carries box velocity, which the removal tracker needs", async () => {
    const seq = await loadSequence("../assets/sequences/nuscenes-scene0.atlasseq");
    const boxes = seq.header.frames[0].boxes;
    expect(boxes[0].velocity).toHaveLength(2);
    let anyMoving = false;
    for (const b of boxes) if (Math.hypot(b.velocity[0], b.velocity[1]) > 0.5) anyMoving = true;
    expect(anyMoving).toBeTruthy();
  });
});
