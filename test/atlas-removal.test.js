import { describe, it, expect } from "./harness.js";
import { configureSensor } from "../src/core/sensor.js";
import { frameOf, atDir } from "./fixtures.js";
import { loadSequence } from "../src/io/sequence.js";
import { buildAttackSchedule } from "../src/core/schedule.js";
import {
  wrapToPi,
  NUSCENES_TRACK,
  ahfrGateProb,
  boxCenterToAzEl,
  boxCenterRange,
  sectorMask,
  clampAngleStep,
  selectInitialTarget,
  matchWithVelocity,
  ahfrRemoval,
  ahfrAim,
  makeAtlasRng,
  AHFR_DEFAULTS,
} from "../src/core/atlas-removal.js";

const DEG = Math.PI / 180;
const sensor = configureSensor("hdl64e");

/** A deterministic rng whose draws we control exactly. */
function fixedRng(uniforms = [], normals = []) {
  let i = 0;
  let j = 0;
  return {
    random: () => (i < uniforms.length ? uniforms[i++] : 0.5),
    normal: () => (j < normals.length ? normals[j++] : 0),
  };
}

describe("ahfrGateProb — the NDSS distance bands, verbatim", () => {
  it("reproduces each band", () => {
    expect(ahfrGateProb(5)).toBeCloseTo(0.38, 1e-12);
    expect(ahfrGateProb(15)).toBeCloseTo(0.66, 1e-12);
    expect(ahfrGateProb(25)).toBeCloseTo(0.46, 1e-12);
    expect(ahfrGateProb(35)).toBeCloseTo(0.82, 1e-12);
    expect(ahfrGateProb(45)).toBeCloseTo(0.93, 1e-12);
    expect(ahfrGateProb(80)).toBeCloseTo(1.0, 1e-12);
  });

  it("resolves band boundaries to the lower band, as the if/elif chain does", () => {
    // ATLAS writes `elif distance <= 20 and distance >= 10`, so an exact 20
    // matches that branch rather than the 20-30 one. Faithfulness means
    // reproducing the boundary behaviour, not tidying it.
    expect(ahfrGateProb(10)).toBeCloseTo(0.66, 1e-12);
    expect(ahfrGateProb(20)).toBeCloseTo(0.66, 1e-12);
    expect(ahfrGateProb(30)).toBeCloseTo(0.46, 1e-12);
    expect(ahfrGateProb(40)).toBeCloseTo(0.82, 1e-12);
    expect(ahfrGateProb(50)).toBeCloseTo(0.93, 1e-12);
  });

  it("is not monotonic in distance, because it comes from measurements", () => {
    // 10-20 m is 0.66 but 20-30 m drops to 0.46. A model that smoothed this
    // would not be the ATLAS model.
    expect(ahfrGateProb(15)).toBeGreaterThan(ahfrGateProb(25));
    expect(ahfrGateProb(35)).toBeGreaterThan(ahfrGateProb(25));
  });
});

describe("angle helpers", () => {
  it("wraps to [-pi, pi]", () => {
    expect(wrapToPi(0)).toBeCloseTo(0, 1e-12);
    expect(wrapToPi(Math.PI + 0.1)).toBeCloseTo(-Math.PI + 0.1, 1e-12);
    expect(wrapToPi(-Math.PI - 0.1)).toBeCloseTo(Math.PI - 0.1, 1e-12);
    // ATLAS's `(a + pi) % (2pi) - pi` maps an exact 3pi to -pi, not +pi
    // (verified against numpy). The half-open convention is worth pinning down,
    // because the sector centre is wrapped with it.
    expect(wrapToPi(3 * Math.PI)).toBeCloseTo(-Math.PI, 1e-9);
    expect(wrapToPi(-3 * Math.PI)).toBeCloseTo(-Math.PI, 1e-9);
  });

  it("clamps an aiming step to the servo rate limit", () => {
    const maxStep = 15 * DEG;
    expect(clampAngleStep(0, 5 * DEG, maxStep)).toBeCloseTo(5 * DEG, 1e-12);
    expect(clampAngleStep(0, 40 * DEG, maxStep)).toBeCloseTo(15 * DEG, 1e-12);
    expect(clampAngleStep(0, -40 * DEG, maxStep)).toBeCloseTo(-15 * DEG, 1e-12);
  });

  it("takes the short way round when clamping across the wrap", () => {
    const maxStep = 15 * DEG;
    const out = clampAngleStep(Math.PI - 0.05, -Math.PI + 0.05, maxStep);
    expect(Math.abs(wrapToPi(out - (Math.PI - 0.05)))).toBeLessThanOrEqual(maxStep + 1e-12);
  });

  it("converts a box centre to the azimuth/elevation the attacker aims at", () => {
    const { az, el } = boxCenterToAzEl([0, 10, 0, 4, 2, 1.6, 0]);
    expect(az).toBeCloseTo(Math.PI / 2, 1e-9);
    expect(el).toBeCloseTo(0, 1e-9);
    expect(boxCenterRange([3, 4, 0, 4, 2, 1.6, 0])).toBeCloseTo(5, 1e-9);
  });

  it("returns azimuth wrapped to [-pi, pi], not [0, 2pi)", () => {
    // ATLAS's removal path uses _wrap_to_pi, unlike its injection path which
    // uses [0, 2pi); mixing them up silently mis-aims the sector.
    const { az } = boxCenterToAzEl([-10, -0.001, 0, 4, 2, 1.6, 0]);
    expect(az).toBeLessThan(0);
    expect(az).toBeGreaterThan(-Math.PI);
  });
});

describe("sectorMask — a rectangular az x el wedge, not a derived envelope", () => {
  const pts = [
    atDir(0, 0, 30),
    atDir(9 * DEG, 0, 30),
    atDir(11 * DEG, 0, 30),
    atDir(0, 7 * DEG, 30),
    atDir(0, 9 * DEG, 30),
  ];

  it("includes points inside the default 20 x 16 degree sector", () => {
    const frame = frameOf(sensor, pts);
    const m = sectorMask(frame, 0, 0, 20, 16);
    expect(m[0]).toBe(1); // on axis
    expect(m[1]).toBe(1); // 9 deg az, inside the 10 deg half-width
  });

  it("excludes points beyond the azimuth half-width", () => {
    const frame = frameOf(sensor, pts);
    const m = sectorMask(frame, 0, 0, 20, 16);
    expect(m[2]).toBe(0); // 11 deg az > 10 deg half-width
  });

  it("excludes points beyond the elevation half-width", () => {
    const frame = frameOf(sensor, pts);
    const m = sectorMask(frame, 0, 0, 20, 16);
    expect(m[3]).toBe(1); // 7 deg el, inside the 8 deg half-width
    expect(m[4]).toBe(0); // 9 deg el > 8 deg half-width
  });

  it("is a hard rectangle: no falloff with angle", () => {
    // This is a real difference from the Cao-mechanism model in remove.js, whose
    // per-cell rate is azimuth-peaked. ATLAS's sector is uniform.
    const frame = frameOf(sensor, [atDir(0, 0, 30), atDir(9.5 * DEG, 0, 30)]);
    const m = sectorMask(frame, 0, 0, 20, 16);
    expect(m[0]).toBe(m[1]);
  });

  it("handles a sector centred across the +-pi wrap", () => {
    const frame = frameOf(sensor, [atDir(Math.PI - 0.02, 0, 30), atDir(-Math.PI + 0.02, 0, 30)]);
    const m = sectorMask(frame, Math.PI, 0, 20, 16);
    expect(m[0]).toBe(1);
    expect(m[1]).toBe(1);
  });

  it("widens with the az_width that names the removal_azNN variants", () => {
    const frame = frameOf(sensor, [atDir(25 * DEG, 0, 30)]);
    expect(sectorMask(frame, 0, 0, 20, 16)[0]).toBe(0);
    expect(sectorMask(frame, 0, 0, 60, 16)[0]).toBe(1);
  });
});

describe("selectInitialTarget", () => {
  // [x, y, z, dx, dy, dz, yaw, class] with class 1=vehicle 2=ped 3=cyclist
  const boxes = [
    [30, 0, 0, 4, 2, 1.6, 0, 1], // vehicle at 30 m, in front
    [10, 0, 0, 4, 2, 1.6, 0, 1], // too close (< min_range 20)
    [-40, 0, 0, 4, 2, 1.6, 0, 1], // behind (x < 0)
    [32, 2, 0, 0.8, 0.8, 1.8, 0, 2], // pedestrian, wrong class
    [45, 0, 0, 4, 2, 1.6, 0, 1], // vehicle at 45 m
  ];

  it("prefers an in-front vehicle beyond min_range", () => {
    const { index, box } = selectInitialTarget(boxes, { desiredRange: 30, tol: 5 });
    expect(index).toBe(0);
    expect(box[7]).toBe(1);
  });

  it("rejects targets behind the sensor and inside min_range", () => {
    const { index } = selectInitialTarget(boxes, { desiredRange: 10, tol: 2 });
    // nothing in the 8-12 m band qualifies, so it falls back to the furthest
    expect([0, 4]).toContain(index);
  });

  it("only considers the requested class", () => {
    const peds = [[32, 2, 0, 0.8, 0.8, 1.8, 0, 2]];
    expect(selectInitialTarget(peds, { targetClass: 1 }).index).toBe(null);
    expect(selectInitialTarget(peds, { targetClass: 2, minRange: 20 }).index).toBe(0);
  });

  it("falls back to the FURTHEST candidate when the band is empty", () => {
    const { index } = selectInitialTarget(boxes, { desiredRange: 100, tol: 1 });
    expect(index).toBe(4); // 45 m is the furthest qualifying vehicle
  });

  it("within the band, picks the candidate furthest from desired_range", () => {
    // Faithful to ATLAS, which uses argmax(|dist - desired|) here. That looks
    // like it was meant to be argmin, but porting the intent rather than the
    // code would silently change which vehicle every removal split targets.
    const band = [
      [29, 0, 0, 4, 2, 1.6, 0, 1], // 1 m from desired
      [34, 0, 0, 4, 2, 1.6, 0, 1], // 4 m from desired -> ATLAS picks this
    ];
    expect(selectInitialTarget(band, { desiredRange: 30, tol: 5 }).index).toBe(1);
  });

  it("returns null for an empty box list", () => {
    expect(selectInitialTarget([], {}).index).toBe(null);
    expect(selectInitialTarget(null, {}).index).toBe(null);
  });

  it("allows targets behind the sensor when negative is set", () => {
    const behind = [[-40, 0, 0, 4, 2, 1.6, 0, 1]];
    expect(selectInitialTarget(behind, {}).index).toBe(null);
    expect(selectInitialTarget(behind, { negative: true }).index).toBe(0);
  });
});

describe("ahfrRemoval", () => {
  const targetBox = [30, 0, 0, 4.2, 1.8, 1.6, 0];

  function sceneAroundTarget() {
    const pts = [];
    // inside the sector, at the target
    for (let i = 0; i < 40; i++) pts.push(atDir((i - 20) * 0.4 * DEG, 0, 30));
    // outside in azimuth
    for (let i = 0; i < 10; i++) pts.push(atDir((30 + i) * DEG, 0, 30));
    // outside in elevation
    for (let i = 0; i < 10; i++) pts.push(atDir(0, (12 + i) * DEG, 30));
    return frameOf(sensor, pts);
  }

  it("removes exactly the sector points when p_remove is 1 and the gate fires", () => {
    const frame = sceneAroundTarget();
    const res = ahfrRemoval({
      frame,
      targetBox,
      pRemove: 1.0,
      rng: fixedRng([0.0]), // gate draw passes
    });
    const expectedMask = sectorMask(frame, 0, 0, AHFR_DEFAULTS.azWidthDeg, AHFR_DEFAULTS.elWidthDeg);
    let removed = 0;
    for (let i = 0; i < frame.count; i++) {
      expect(res.keepMask[i]).toBe(expectedMask[i] ? 0 : 1);
      if (!res.keepMask[i]) removed++;
    }
    expect(res.meta.nPointsRemoved).toBe(removed);
    expect(res.meta.activeThisFrame).toBeTruthy();
  });

  it("removes nothing when the frame-level scan gate does not fire", () => {
    const frame = sceneAroundTarget();
    const res = ahfrRemoval({ frame, targetBox, pRemove: 1.0, rng: fixedRng([0.999]) });
    expect(res.meta.activeThisFrame).toBeFalsy();
    expect(res.meta.inactiveReason).toBe("scan_gate");
    expect(res.meta.nPointsRemoved).toBe(0);
  });

  it("refuses a target inside min_target_range and says why", () => {
    const res = ahfrRemoval({
      frame: sceneAroundTarget(),
      targetBox: [4, 0, 0, 4.2, 1.8, 1.6, 0],
      pRemove: 1.0,
      rng: fixedRng([0.0]),
    });
    expect(res.meta.inactiveReason).toBe("target_too_close");
    expect(res.meta.rangeGatePassed).toBeFalsy();
    expect(res.meta.nPointsRemoved).toBe(0);
  });

  it("allows a close target when extrapolation is explicitly enabled", () => {
    const res = ahfrRemoval({
      frame: sceneAroundTarget(),
      targetBox: [4, 0, 0, 4.2, 1.8, 1.6, 0],
      pRemove: 1.0,
      allowCloseRangeExtrapolation: true,
      rng: fixedRng([0.0]),
    });
    expect(res.meta.inactiveReason).not.toBe("target_too_close");
    expect(res.meta.extrapolatedCloseRange).toBeTruthy();
  });

  it("reports an empty sector rather than pretending to succeed", () => {
    const frame = frameOf(sensor, [atDir(90 * DEG, 0, 30)]);
    const res = ahfrRemoval({ frame, targetBox, pRemove: 1.0, rng: fixedRng([0.0]) });
    expect(res.meta.inactiveReason).toBe("empty_sector");
    expect(res.meta.nPointsInSector).toBe(0);
  });

  it("removes about p_remove of the sector on average", () => {
    const frame = sceneAroundTarget();
    const res = ahfrRemoval({
      frame,
      targetBox,
      pRemove: 0.5,
      rng: makeAtlasRng(1234),
      forceActive: true,
    });
    const frac = res.meta.nPointsRemoved / res.meta.nPointsInSector;
    expect(frac).toBeGreaterThan(0.25);
    expect(frac).toBeLessThan(0.75);
  });

  it("reconciles its bookkeeping", () => {
    const frame = sceneAroundTarget();
    const res = ahfrRemoval({ frame, targetBox, pRemove: 0.96, rng: makeAtlasRng(7), forceActive: true });
    const m = res.meta;
    expect(m.nPointsBefore).toBe(frame.count);
    expect(m.nPointsAfter).toBe(m.nPointsBefore - m.nPointsRemoved);
    expect(m.nPointsRemoved).toBeLessThanOrEqual(m.nPointsInSector);
  });

  it("aims at the target box centre on the first frame", () => {
    const res = ahfrRemoval({
      frame: sceneAroundTarget(),
      targetBox: [0, 30, 0, 4.2, 1.8, 1.6, 0],
      pRemove: 1.0,
      rng: fixedRng([0.0], [0]),
    });
    expect(res.state.azCenter).toBeCloseTo(Math.PI / 2, 1e-6);
  });

  it("rate-limits aim changes between frames, like a servo", () => {
    const frame = sceneAroundTarget();
    const first = ahfrRemoval({ frame, targetBox: [30, 0, 0, 4, 2, 1.6, 0], pRemove: 1.0, rng: fixedRng([0.0], [0]) });
    // target jumps 40 degrees; the aim may only move maxAzStepDeg
    const second = ahfrRemoval({
      frame,
      targetBox: [30 * Math.cos(40 * DEG), 30 * Math.sin(40 * DEG), 0, 4, 2, 1.6, 0],
      state: first.state,
      pRemove: 1.0,
      maxAzStepDeg: 15,
      rng: fixedRng([0.0], [0]),
    });
    expect(second.state.azCenter).toBeCloseTo(15 * DEG, 1e-6);
    expect(second.meta.rawTargetAzRad).toBeCloseTo(40 * DEG, 1e-6);
  });

  it("holds elevation fixed across frames by default", () => {
    const frame = sceneAroundTarget();
    const first = ahfrRemoval({ frame, targetBox: [30, 0, 0, 4, 2, 1.6, 0], pRemove: 1.0, rng: fixedRng([0.0], [0]) });
    const raised = [30, 0, 10, 4, 2, 1.6, 0];
    const second = ahfrRemoval({ frame, targetBox: raised, state: first.state, pRemove: 1.0, rng: fixedRng([0.0], [0]) });
    expect(second.state.elCenter).toBeCloseTo(first.state.elCenter, 1e-12);

    const tracking = ahfrRemoval({
      frame, targetBox: raised, state: first.state, pRemove: 1.0,
      fixElevation: false, rng: fixedRng([0.0], [0]),
    });
    expect(tracking.state.elCenter).not.toBe(first.state.elCenter);
  });

  it("applies aim jitter in degrees, as ATLAS does", () => {
    const frame = sceneAroundTarget();
    const res = ahfrRemoval({
      frame, targetBox: [30, 0, 0, 4, 2, 1.6, 0], pRemove: 1.0,
      centerJitterStdDeg: 1.0,
      rng: fixedRng([0.0], [2.0]), // a +2 sigma draw => +2 degrees
    });
    expect(res.state.azCenter).toBeCloseTo(2 * DEG, 1e-6);
  });

  it("uses the distance-banded gate probability for the target's range", () => {
    const frame = sceneAroundTarget();
    const res = ahfrRemoval({ frame, targetBox: [45, 0, 0, 4, 2, 1.6, 0], pRemove: 1.0, rng: fixedRng([0.9]) });
    expect(res.meta.scanGateProb).toBeCloseTo(0.93, 1e-12);
    expect(res.meta.activeThisFrame).toBeTruthy(); // 0.9 < 0.93
    const res2 = ahfrRemoval({ frame, targetBox: [25, 0, 0, 4, 2, 1.6, 0], pRemove: 1.0, rng: fixedRng([0.9]) });
    expect(res2.meta.scanGateProb).toBeCloseTo(0.46, 1e-12);
    expect(res2.meta.activeThisFrame).toBeFalsy(); // 0.9 > 0.46
  });

  it("exposes ATLAS's documented defaults", () => {
    expect(AHFR_DEFAULTS.azWidthDeg).toBe(20);
    expect(AHFR_DEFAULTS.elWidthDeg).toBe(16);
    expect(AHFR_DEFAULTS.pRemove).toBeCloseTo(0.96, 1e-12);
    expect(AHFR_DEFAULTS.maxAzStepDeg).toBe(15);
    expect(AHFR_DEFAULTS.minTargetRangeM).toBe(6);
    expect(AHFR_DEFAULTS.centerJitterStdDeg).toBeCloseTo(0.05, 1e-12);
    expect(AHFR_DEFAULTS.fixElevation).toBeTruthy();
  });
});

describe("matchWithVelocity — port of ATLAS's constant-velocity tracker", () => {
  // boxes are [x,y,z,dx,dy,dz,yaw,vx,vy,class] - velocity in cols 7,8 and the
  // class in the LAST column, which is what ATLAS indexes
  const mk = (x, y, vx = 0, vy = 0, cls = 1, dx = 4.2, dy = 1.8, dz = 1.6) =>
    [x, y, 0, dx, dy, dz, 0, vx, vy, cls];

  it("matches the candidate nearest the constant-velocity prediction", () => {
    const prev = mk(20, 0, 10, 0); // moving +10 m/s in x
    // at dt = 0.5 the prediction is x = 25
    const curr = [mk(21, 0), mk(25.2, 0), mk(30, 0)];
    const m = matchWithVelocity(prev, null, curr, { dt: 0.5 });
    expect(m.index).toBe(1);
    expect(m.dist).toBeLessThan(1);
  });

  it("uses dt: nuScenes keyframes are 2 Hz, and Waymo's 0.1 loses the track", () => {
    // The trap the docstring calls out. A vehicle at 10 m/s moves 5 m between
    // nuScenes keyframes; predicting only 1 m ahead leaves a 4 m residual, and a
    // fast enough target then falls outside max_step and is dropped silently.
    const prev = mk(20, 0, 24, 0); // 24 m/s
    const curr = [mk(32, 0)]; // 12 m ahead: exactly where dt=0.5 predicts
    expect(matchWithVelocity(prev, null, curr, { dt: 0.5 }).index).toBe(0);
    expect(matchWithVelocity(prev, null, curr, { dt: 0.1 }).index).toBe(null);
  });

  it("rejects candidates beyond the spatial gate", () => {
    const prev = mk(20, 0);
    expect(matchWithVelocity(prev, null, [mk(40, 0)], { dt: 0.5, maxStep: 6 }).index).toBe(null);
    expect(matchWithVelocity(prev, null, [mk(24, 0)], { dt: 0.5, maxStep: 6 }).index).toBe(0);
  });

  it("rejects a candidate whose volume differs too much", () => {
    const prev = mk(20, 0);
    // a bus where a car used to be, 1 m away: inside the spatial gate but wrong size
    const bus = mk(21, 0, 0, 0, 1, 12, 3, 3.5);
    expect(matchWithVelocity(prev, null, [bus], { dt: 0.5, maxVolumeRatio: 2 }).index).toBe(null);
    // and a similarly sized car matches
    expect(matchWithVelocity(prev, null, [mk(21, 0)], { dt: 0.5 }).index).toBe(0);
  });

  it("only considers the requested class", () => {
    const prev = mk(20, 0);
    const ped = mk(20.5, 0, 0, 0, 2, 0.8, 0.8, 1.8);
    expect(matchWithVelocity(prev, null, [ped], { dt: 0.5, targetClass: 1 }).index).toBe(null);
  });

  it("returns nothing when there are no candidates at all", () => {
    expect(matchWithVelocity(mk(20, 0), null, [], { dt: 0.5 }).index).toBe(null);
    expect(matchWithVelocity(mk(20, 0), null, null, { dt: 0.5 }).index).toBe(null);
  });

  it("prefers the nearest of several plausible candidates", () => {
    const prev = mk(20, 0);
    const curr = [mk(23, 0), mk(21, 0), mk(24, 0)];
    expect(matchWithVelocity(prev, null, curr, { dt: 0.5 }).index).toBe(1);
  });

  it("returns the matched box itself so the caller can carry state forward", () => {
    const prev = mk(20, 0, 4, 0);
    const curr = [mk(22, 0.5, 4, 0)];
    const m = matchWithVelocity(prev, null, curr, { dt: 0.5 });
    expect(m.box[0]).toBeCloseTo(22, 1e-9);
    expect(m.box[1]).toBeCloseTo(0.5, 1e-9);
  });
});

describe("ahfrAim", () => {
  const tb = [30, 5, 0, 4.2, 1.8, 1.6, 0];

  it("agrees with the aim ahfrRemoval computes for the same seed and state", () => {
    // This is what lets a sequence propagate aim state cheaply: running the aim
    // alone must land exactly where the full removal would.
    for (const seed of [1, 5, 99]) {
      const aim = ahfrAim({ targetBox: tb, rng: makeAtlasRng(seed) });
      const full = ahfrRemoval({
        frame: frameOf(sensor, [atDir(0, 0, 30)]),
        targetBox: tb,
        rng: makeAtlasRng(seed),
      });
      expect(aim.azCenter).toBeCloseTo(full.meta.azCenterRad, 1e-12);
      expect(aim.elCenter).toBeCloseTo(full.meta.elCenterRad, 1e-12);
      expect(aim.scanGateProb).toBeCloseTo(full.meta.scanGateProb, 1e-12);
    }
  });

  // END-TO-END OVER A REAL SEQUENCE
  // Every other tracker test uses a synthetic box path. This one runs ATLAS's
  // actual loop - selectInitialTarget on frame 0, then match_with_velocity frame
  // to frame, with aim state threaded through - over a real 40-keyframe nuScenes
  // scene. It only became worth writing once the exported sequences were full
  // length: on a 20-frame export the track had barely enough frames to be tested
  // against, and every intermittent schedule collapsed to degenerate.
  it("tracks a real target across a full nuScenes sequence", async () => {
    const seq = await loadSequence("../assets/sequences/nuscenes-scene0.atlasseq");
    const dt = seq.header.dtSec;
    const frames = seq.header.frames;

    const asAtlas = (boxes) =>
      boxes.map((b) => [...b.center, ...b.size, b.yaw, ...b.velocity,
                        /car|van|truck|bus|trailer|vehicle/i.test(b.label) ? 1 : 2]);

    let prev = null;
    let state = null;
    let held = 0;
    for (let k = 0; k < frames.length; k++) {
      const boxes = asAtlas(frames[k].boxes);
      const box = k === 0
        ? selectInitialTarget(boxes, {
            desiredRange: NUSCENES_TRACK.desiredRange, tol: AHFR_DEFAULTS.tol,
            targetClass: 1, minRange: NUSCENES_TRACK.minRange,
            negative: NUSCENES_TRACK.negative,
          }).box
        : matchWithVelocity(prev, null, boxes, { dt, maxStep: NUSCENES_TRACK.maxStep }).box;
      if (!box) break; // ATLAS gives up for the rest of the sequence
      const aim = ahfrAim({ targetBox: box, state, rng: makeAtlasRng(7 + k) });
      state = aim.state;
      prev = box;
      held++;
      expect(Number.isFinite(aim.azCenter)).toBeTruthy();
    }
    // The track must survive well past the 4-frame clean prefix, or the temporal
    // half of the attack is untestable. It need not survive to the end: a target
    // can genuinely leave the scene, and ATLAS then stops for good.
    expect(held).toBeGreaterThan(10);
  });

  it("gives an intermittent schedule real gaps to track through", async () => {
    const seq = await loadSequence("../assets/sequences/nuscenes-scene0.atlasseq");
    const s = buildAttackSchedule({
      nFrames: seq.header.nFrames, seed: 42, mode: "intermittent", label: seq.header.slug,
    });
    expect(s.meta.degenerate).toBeFalsy();
    expect(s.meta.nGaps).toBeGreaterThan(0);
    // a gap means at least one clean frame strictly inside the attacked span
    const first = s.attacked[0];
    const last = s.attacked[s.attacked.length - 1];
    expect(s.meta.gapLocal.every((g) => g > first && g < last)).toBeTruthy();
  });

  it("propagates state across frames with the servo limit applied", () => {
    let state = null;
    const aims = [];
    // target jumps 40 degrees each frame; the aim may only move 15
    for (let k = 0; k < 4; k++) {
      const ang = k * 40 * DEG;
      const box = [30 * Math.cos(ang), 30 * Math.sin(ang), 0, 4.2, 1.8, 1.6, 0];
      const a = ahfrAim({ targetBox: box, state, rng: makeAtlasRng(1 + k) });
      aims.push(a.azCenter);
      state = a.state;
    }
    // ATLAS adds the aim jitter AFTER the servo clamp, so a realized step can
    // exceed max_az_step_deg by a few jitter sigma (0.05 deg each, on both this
    // frame's and the previous frame's draw). Allow that rather than pretending
    // the clamp is a hard bound on the observed motion.
    const slack = 15 * DEG + 6 * 0.05 * DEG;
    for (let k = 1; k < aims.length; k++) {
      expect(Math.abs(wrapToPi(aims[k] - aims[k - 1]))).toBeLessThanOrEqual(slack);
    }
  });
});
