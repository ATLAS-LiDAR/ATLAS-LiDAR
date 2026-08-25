import { describe, it, expect } from "./harness.js";
import { configureSensor } from "../src/core/sensor.js";
import { configureSpoofer } from "../src/core/spoofer.js";
import {
  victimApertureRad,
  attackerCanIlluminate,
  illuminationDuty,
  illuminationWindow,
  ATTACKER_MODES,
  attackerEgoPosition,
  worldFromEgoPoint,
  sectorCoverage,
  attackerGeometry,
} from "../src/core/attacker.js";
import { applyPose, invertRigid4 } from "../src/core/pose.js";

const DEG = Math.PI / 180;
const sensor = configureSensor("hdl32e");

/** ego->global: rotation about z then translation. */
const pose = (yaw, tx, ty) => {
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [c, -s, 0, tx, s, c, 0, ty, 0, 0, 1, 0, 0, 0, 0, 1];
};

describe("attacker placement modes", () => {
  it("offers a static roadside attacker and a dynamic one that follows the ego", () => {
    expect(ATTACKER_MODES).toContain("static");
    expect(ATTACKER_MODES).toContain("dynamic");
  });

  it("a STATIC attacker holds one world position while the ego drives past", () => {
    // It is a box on the ground. Its ego-relative bearing and range must both
    // change every frame - approaching, passing, receding - which is the whole
    // point: it is not welded to the vehicle.
    const worldPos = [50, 0, 0];
    const ranges = [];
    const bearings = [];
    for (const tx of [0, 20, 40, 60, 80]) {
      const p = attackerEgoPosition({ mode: "static", worldPos, pose: pose(0, tx, 0) });
      ranges.push(Math.hypot(p[0], p[1]));
      bearings.push(Math.atan2(p[1], p[0]));
    }
    // driving toward it then past it: range falls to ~0 then grows again
    expect(ranges[0]).toBeCloseTo(50, 1e-6);
    expect(ranges[1]).toBeLessThan(ranges[0]);
    expect(ranges[2]).toBeLessThan(ranges[1]);
    expect(ranges[4]).toBeGreaterThan(ranges[2]);
    // and it ends up behind the ego
    expect(bearings[4]).toBeCloseTo(Math.PI, 1e-6);
  });

  it("a static attacker's world position is unchanged by the ego moving", () => {
    const worldPos = [30, 12, 0];
    for (const tx of [0, 15, 45]) {
      const p = pose(0.3, tx, 0);
      const ego = attackerEgoPosition({ mode: "static", worldPos, pose: p });
      // lifting the ego-frame position back into the world recovers the original
      expect(applyPose(p, ego)).toBeCloseToArray(worldPos, 1e-6);
    }
  });

  it("a DYNAMIC attacker keeps a fixed offset from the ego, so it travels along", () => {
    const egoOffset = [18, 4, 0];
    for (const tx of [0, 25, 90]) {
      const p = attackerEgoPosition({ mode: "dynamic", egoOffset, pose: pose(0.9, tx, 0) });
      expect(p).toBeCloseToArray(egoOffset, 1e-9);
    }
  });

  it("converts an ego-frame click into the world position a static attacker keeps", () => {
    const p = pose(0.7, 100, -40);
    const world = worldFromEgoPoint([20, 5, 0], p);
    expect(attackerEgoPosition({ mode: "static", worldPos: world, pose: p })).toBeCloseToArray([20, 5, 0], 1e-6);
  });

  it("falls back to the ego offset when a static attacker has no world position yet", () => {
    const p = attackerEgoPosition({ mode: "static", worldPos: null, egoOffset: [12, 0, 0], pose: pose(0, 5, 5) });
    expect(p).toBeCloseToArray([12, 0, 0], 1e-9);
  });

  it("works without a pose, as on a static single frame", () => {
    const p = attackerEgoPosition({ mode: "static", worldPos: [22, 3, 0], pose: null, egoOffset: [9, 1, 0] });
    expect(p).toBeCloseToArray([9, 1, 0], 1e-9);
  });
});

describe("sectorCoverage — containment, not alignment", () => {
  // An attacker does NOT have to sit on the azimuth it wants to inject at. It can
  // reach a band around its own bearing, whose half-width comes from its power
  // over the victim's detection threshold. So the criterion is whether the
  // attacked sector falls INSIDE that band.
  const cover = (offsetDeg, sectorDeg, envDeg) =>
    sectorCoverage({
      attackerBearing: 0,
      sectorCenterAz: offsetDeg * DEG,
      sectorWidthRad: sectorDeg * DEG,
      envelopeHalfWidthRad: envDeg * DEG,
    });

  it("fully covers a sector well inside the envelope", () => {
    const c = cover(0, 10, 20);
    expect(c.fraction).toBeCloseTo(1, 1e-9);
    expect(c.withinEnvelope).toBeTruthy();
    expect(c.offsetDeg).toBeCloseTo(0, 1e-9);
  });

  it("still fully covers an OFF-AXIS sector, as long as it fits in the band", () => {
    // the key point being pushed back on: 15 deg off-axis is fine for a 20 deg
    // half-width envelope
    const c = cover(15, 8, 20);
    expect(c.fraction).toBeCloseTo(1, 1e-9);
    expect(c.withinEnvelope).toBeTruthy();
    expect(c.offsetDeg).toBeCloseTo(15, 1e-6);
  });

  it("partially covers a sector straddling the envelope edge", () => {
    // sector spans 16..24 deg, envelope reaches 20 deg -> half of it reachable
    const c = cover(20, 8, 20);
    expect(c.fraction).toBeCloseTo(0.5, 1e-6);
    expect(c.withinEnvelope).toBeFalsy();
    expect(c.partial).toBeTruthy();
  });

  it("covers nothing once the sector is entirely outside the envelope", () => {
    const c = cover(40, 8, 20);
    expect(c.fraction).toBeCloseTo(0, 1e-9);
    expect(c.withinEnvelope).toBeFalsy();
    expect(c.partial).toBeFalsy();
  });

  it("is symmetric in the sign of the offset", () => {
    expect(cover(18, 10, 20).fraction).toBeCloseTo(cover(-18, 10, 20).fraction, 1e-12);
  });

  it("handles the wrap, so an offset near +-180 is measured the short way", () => {
    const c = sectorCoverage({
      attackerBearing: 0.02,
      sectorCenterAz: 2 * Math.PI - 0.02,
      sectorWidthRad: 4 * DEG,
      envelopeHalfWidthRad: 10 * DEG,
    });
    expect(Math.abs(c.offsetDeg)).toBeLessThan(3);
    expect(c.fraction).toBeCloseTo(1, 1e-9);
  });

  it("reports zero coverage for a zero-width envelope", () => {
    expect(cover(0, 10, 0).fraction).toBeCloseTo(0, 1e-9);
  });
});

describe("attackerGeometry", () => {
  it("combines placement, envelope and coverage into one verdict", () => {
    const g = attackerGeometry({
      mode: "static",
      worldPos: [40, 0, 0],
      pose: pose(0, 0, 0),
      sensor,
      spoofer: configureSpoofer("cao2023"),
      sectorCenterAz: 0,
      sectorWidthRad: 10 * DEG,
    });
    expect(g.egoPos).toHaveLength(3);
    expect(g.standoff).toBeCloseTo(40, 1e-6);
    expect(g.envelopeHalfWidthDeg).toBeGreaterThan(1);
    expect(g.coverage.fraction).toBeGreaterThan(0);
  });

  it("widens the envelope as the ego closes on a static attacker", () => {
    // received power goes as 1/d^2, so the reachable band opens up as the car
    // approaches and narrows again as it recedes - a window of opportunity
    const common = {
      mode: "static", worldPos: [60, 0, 0], sensor,
      spoofer: configureSpoofer("cao2023"), sectorCenterAz: 0, sectorWidthRad: 10 * DEG,
    };
    const far = attackerGeometry({ ...common, pose: pose(0, 0, 0) });
    const near = attackerGeometry({ ...common, pose: pose(0, 45, 0) });
    expect(near.standoff).toBeLessThan(far.standoff);
    expect(near.envelopeHalfWidthDeg).toBeGreaterThan(far.envelopeHalfWidthDeg);
  });

  it("loses coverage as a static attacker is left behind", () => {
    const common = {
      mode: "static", worldPos: [40, 0, 0], sensor,
      spoofer: configureSpoofer("cao2023"), sectorWidthRad: 10 * DEG,
    };
    // the attack keeps aiming straight ahead while the attacker falls behind
    const early = attackerGeometry({ ...common, pose: pose(0, 0, 0), sectorCenterAz: 0 });
    const late = attackerGeometry({ ...common, pose: pose(0, 70, 0), sectorCenterAz: 0 });
    // On axis, but only ~0.68 covered: the sector (10 deg) is WIDER than the
    // reference-power envelope (6.8 deg), so perfect aim is not enough.
    expect(early.coverage.fraction).toBeCloseTo(0.68, 0.02);
    expect(early.coverage.offsetDeg).toBeCloseTo(0, 1e-6);
    expect(Math.abs(late.coverage.offsetDeg)).toBeGreaterThan(Math.abs(early.coverage.offsetDeg));
    expect(late.coverage.fraction).toBeLessThan(early.coverage.fraction);
    expect(late.coverage.fraction).toBeCloseTo(0, 1e-9); // driven clean past it
  });

  it("the reference-power envelope is NARROW, so even modest drift costs coverage", () => {
    // Worth pinning: the measured Cao envelope is 6.8 deg FULL width, i.e. ~3.4 deg
    // either side. Off-axis injection is possible, but a wide off-axis attack needs
    // real power (or a much closer standoff), not just intent.
    const g = attackerGeometry({
      mode: "static", worldPos: [40 * Math.cos(14 * DEG), 40 * Math.sin(14 * DEG), 0],
      pose: pose(0, 0, 0), sensor, spoofer: configureSpoofer("cao2023"),
      sectorCenterAz: 0, sectorWidthRad: 10 * DEG,
    });
    expect(g.envelopeHalfWidthDeg).toBeLessThan(5);
    // sign is "sector relative to attacker"; magnitude is what matters
    expect(Math.abs(g.coverage.offsetDeg)).toBeCloseTo(14, 0.5);
    expect(g.coverage.fraction).toBeCloseTo(0, 1e-9);

    // raise the power and the same geometry becomes reachable
    const strong = attackerGeometry({
      mode: "static", worldPos: [40 * Math.cos(14 * DEG), 40 * Math.sin(14 * DEG), 0],
      pose: pose(0, 0, 0), sensor,
      spoofer: configureSpoofer("cao2023", { powerRatio: 5000 }),
      sectorCenterAz: 0, sectorWidthRad: 10 * DEG,
    });
    expect(strong.envelopeHalfWidthDeg).toBeGreaterThan(g.envelopeHalfWidthDeg);
    expect(strong.coverage.fraction).toBeGreaterThan(0);
  });

  it("a dynamic attacker holds its geometry as the ego moves", () => {
    const common = {
      mode: "dynamic", egoOffset: [18, 0, 0], sensor,
      spoofer: configureSpoofer("cao2023"), sectorCenterAz: 0, sectorWidthRad: 10 * DEG,
    };
    const a = attackerGeometry({ ...common, pose: pose(0, 0, 0) });
    const b = attackerGeometry({ ...common, pose: pose(0.4, 55, -20) });
    expect(b.standoff).toBeCloseTo(a.standoff, 1e-9);
    expect(b.coverage.fraction).toBeCloseTo(a.coverage.fraction, 1e-9);
  });
});

describe("ATLAS sector widths vs what a real box can reach", () => {
  // A realizability observation the drift-honest view makes unavoidable: ATLAS's
  // DEFAULT removal sector is 20 deg, roughly 3x the 6.8 deg envelope measured on
  // real hardware at the reference power. So even a perfectly aimed roadside box
  // covers only about a third of it, and the wider removal_azNN splits are
  // further out still - consistent with ATLAS_REAL_WORLD_VALIDATION.md labelling
  // az50/az60 capability extrapolations.
  const coverFor = (sectorDeg, powerRatio = undefined) =>
    attackerGeometry({
      mode: "static", worldPos: [40, 0, 0], pose: pose(0, 0, 0), sensor,
      spoofer: configureSpoofer("cao2023", powerRatio ? { powerRatio } : {}),
      sectorCenterAz: 0, sectorWidthRad: sectorDeg * DEG,
    }).coverage.fraction;

  it("covers a narrow sector fully at reference power", () => {
    expect(coverFor(5)).toBeCloseTo(1, 1e-9);
  });

  it("covers only a fraction of ATLAS's default 20 degree removal sector", () => {
    const f = coverFor(20);
    expect(f).toBeGreaterThan(0.2);
    expect(f).toBeLessThan(0.45);
  });

  it("covers even less of the widest removal_az60 split", () => {
    expect(coverFor(60)).toBeLessThan(coverFor(20));
  });

  it("needs more power, not better aim, to cover a wide sector", () => {
    expect(coverFor(20, 1e5)).toBeGreaterThan(coverFor(20));
  });
});

describe("victim aperture and when the attacker can actually get in", () => {
  it("reports the victim's instantaneous azimuthal acceptance", () => {
    // A receiver co-boresighted with its emitter accepts light from a narrow cone
    // around the current beam direction - a few mrad, not degrees.
    const a = victimApertureRad(sensor);
    expect(a).toBeCloseTo(sensor.beamDivergenceMrad / 1000, 1e-12);
    expect(a / DEG).toBeLessThan(0.5);
    expect(a).toBeGreaterThan(0);
  });

  it("lets the attacker in only while the aperture is inside its band", () => {
    const band = 5 * DEG;
    const ap = victimApertureRad(sensor);
    const can = (sweepDeg) =>
      attackerCanIlluminate({
        sweepAz: sweepDeg * DEG,
        attackerBearing: 30 * DEG,
        envelopeHalfWidthRad: band,
        apertureRad: ap,
      });
    expect(can(30)).toBeTruthy();   // dead on
    expect(can(34)).toBeTruthy();   // inside the band
    expect(can(36)).toBeFalsy();    // outside it
    expect(can(0)).toBeFalsy();     // pointed elsewhere entirely
    expect(can(210)).toBeFalsy();   // pointed the opposite way
  });

  it("keys off the ATTACKER's bearing, not the attacked sector's", () => {
    // The bug this fixes: with a world-fixed attacker the two diverge, and firing
    // must follow where the hardware actually stands.
    const ap = victimApertureRad(sensor);
    const opts = { attackerBearing: 80 * DEG, envelopeHalfWidthRad: 4 * DEG, apertureRad: ap };
    expect(attackerCanIlluminate({ sweepAz: 80 * DEG, ...opts })).toBeTruthy();
    // the sector might be aimed at 0, but that is irrelevant to getting in
    expect(attackerCanIlluminate({ sweepAz: 0, ...opts })).toBeFalsy();
  });

  it("handles the wrap past zero", () => {
    const ap = victimApertureRad(sensor);
    const opts = { attackerBearing: 2 * Math.PI - 0.01, envelopeHalfWidthRad: 3 * DEG, apertureRad: ap };
    expect(attackerCanIlluminate({ sweepAz: 0.005, ...opts })).toBeTruthy();
    expect(attackerCanIlluminate({ sweepAz: Math.PI, ...opts })).toBeFalsy();
  });

  it("is a tiny fraction of each revolution", () => {
    // The window is the band plus one aperture, so a 5 deg band is ~1.4% of a turn.
    // That brevity is the point: the attacker is shut out almost all the time.
    const duty = illuminationDuty({ envelopeHalfWidthRad: 5 * DEG, apertureRad: victimApertureRad(sensor) });
    expect(duty).toBeGreaterThan(0.01);
    expect(duty).toBeLessThan(0.04);
  });

  it("widens the duty with the band and shrinks it to nothing without one", () => {
    const ap = victimApertureRad(sensor);
    const wide = illuminationDuty({ envelopeHalfWidthRad: 20 * DEG, apertureRad: ap });
    const narrow = illuminationDuty({ envelopeHalfWidthRad: 2 * DEG, apertureRad: ap });
    expect(wide).toBeGreaterThan(narrow);
    // with no reachable band at all, only the bare aperture lets anything in
    const none = illuminationDuty({ envelopeHalfWidthRad: 0, apertureRad: ap });
    expect(none).toBeCloseTo(ap / (2 * Math.PI), 1e-12);
  });

  it("gives the arc over which illumination is possible, for drawing", () => {
    const w = illuminationWindow({
      attackerBearing: 30 * DEG,
      envelopeHalfWidthRad: 5 * DEG,
      apertureRad: victimApertureRad(sensor),
    });
    expect(w.centerAz).toBeCloseTo(30 * DEG, 1e-12);
    expect(w.halfWidthRad).toBeGreaterThan(5 * DEG);
    expect(w.halfWidthRad).toBeCloseTo(5 * DEG + victimApertureRad(sensor) / 2, 1e-12);
  });
});
