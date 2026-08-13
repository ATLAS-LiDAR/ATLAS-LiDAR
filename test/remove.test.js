import { describe, it, expect } from "./harness.js";
import { configureSensor } from "../src/core/sensor.js";
import { configureSpoofer, CAO2023_ENVELOPE } from "../src/core/spoofer.js";
import { frameOf, atCell, wallFrame } from "./fixtures.js";
import { unpackCellKey, cellKey } from "../src/core/polar.js";
import { removeReturns, OUTCOMES } from "../src/core/remove.js";

const DEG = Math.PI / 180;
const vlp16 = configureSensor("vlp16");
const SPOOF_POS = [40, 0, 0];

/**
 * A surface filling every cell across the whole envelope, at constant range.
 * It must sit BEYOND the spoofer: a surface between victim and spoofer would
 * block the attacker's line of sight, which checkLineOfSight correctly vetoes.
 */
function surface(sensor, range = 60) {
  return wallFrame(sensor, { azFromDeg: -12, azToDeg: 12, range });
}

function attack(opts) {
  return removeReturns({
    sensor: vlp16,
    spoofer: configureSpoofer("cao2023"),
    positionXYZ: SPOOF_POS,
    attackerRangeM: 0.2, // inside the VLP-16's 40 cm MOT
    seed: 5,
    ...opts,
  });
}

describe("removal via sub-MOT saturation", () => {
  it("destroys genuine returns when the attacker echo lands inside the MOT", () => {
    // The real mechanism: the attacker injects a saturated echo below the
    // Minimum Operational Threshold. The sensor prefers the strongest echo and
    // firmware discards sub-MOT returns, so the genuine return is destroyed and
    // the attacker's own point never appears either.
    const frame = surface(vlp16);
    const res = attack({ frame });
    expect(res.stats.removed).toBeGreaterThan(0);
    expect(res.stats.outcome).toBe(OUTCOMES.REMOVED);
    let zeroed = 0;
    for (let i = 0; i < res.keepMask.length; i++) if (!res.keepMask[i]) zeroed++;
    expect(zeroed).toBe(res.stats.removed);
  });

  it("reproduces the measured aim-point removal rate of 0.83-0.87", () => {
    const frame = surface(vlp16);
    const res = attack({ frame });
    expect(res.stats.peakRate).toBeGreaterThan(0.70);
    expect(res.stats.peakRate).toBeLessThan(0.95);
  });

  it("is azimuth-peaked, not flat across the envelope", () => {
    // ATLAS_REAL_WORLD_VALIDATION.md: averaging over the envelope understates
    // the rate ~2x, because per-cell rate runs 0.75 at the aim point down to
    // 0.25 at the fringe. A flat model gets this qualitatively wrong.
    const frame = surface(vlp16);
    const res = attack({ frame });
    expect(res.stats.peakRate).toBeGreaterThan(res.stats.edgeRate * 1.8);
    expect(res.stats.envelopeMeanRate).toBeLessThan(res.stats.peakRate);
  });

  it("lands the fringe rate near the measured 0.25", () => {
    const frame = surface(vlp16);
    const res = attack({ frame });
    expect(res.stats.edgeRate).toBeGreaterThan(0.10);
    expect(res.stats.edgeRate).toBeLessThan(0.45);
    expect(CAO2023_ENVELOPE.edgeRemovalRate).toBe(0.25);
  });

  it("spans the full vertical field of view, as the real capture does", () => {
    const frame = surface(vlp16);
    const res = attack({ frame });
    const rows = new Set();
    for (const k of res.stats.affectedCells) {
      rows.add(unpackCellKey(k, vlp16.nBeams).beamIdx);
    }
    expect(rows.size).toBe(CAO2023_ENVELOPE.channelsAffected);
  });

  it("stays surgical: it deletes a small fraction of the whole cloud", () => {
    // Measured: ATLAS removal deletes 0.6-4.2% of the cloud, versus 50-90% for
    // weather corruption baselines. Removal is targeted, not global.
    const frame = wallFrame(vlp16, { azFromDeg: -180, azToDeg: 179, range: 60 });
    const res = attack({ frame });
    const frac = res.stats.removed / frame.count;
    expect(frac).toBeGreaterThan(0);
    expect(frac).toBeLessThan(0.10);
  });

  it("leaves cells outside the envelope untouched", () => {
    const frame = surface(vlp16);
    const res = attack({ frame });
    const affected = new Set(Array.from(res.stats.affectedCells));
    for (let i = 0; i < frame.count; i++) {
      if (!res.keepMask[i]) expect(affected.has(frame.cell[i])).toBeTruthy();
    }
    // a point 90 degrees away always survives
    const behind = frameOf(vlp16, [
      atCell(vlp16, 0, 8, 25),
      atCell(vlp16, Math.round(Math.PI / 2 / vlp16.azRes), 8, 25),
    ]);
    const res2 = attack({ frame: behind });
    expect(res2.keepMask[1]).toBe(1);
  });
});

describe("removal requires pulse prediction", () => {
  it("fails entirely for a spoofer that cannot fire before the victim's pulse", () => {
    // Derived, not assumed: a sub-MOT echo must arrive within ~2.7 ns of the
    // victim firing, so a spoofer 40 m away has to predict the next pulse. A
    // purely reactive spoofer physically cannot perform this attack.
    const frame = surface(vlp16);
    const res = attack({ frame, spoofer: configureSpoofer("cao2023", { canPredict: false }) });
    expect(res.stats.removed).toBe(0);
    expect(res.stats.outcome).toBe(OUTCOMES.INFEASIBLE);
    expect(res.stats.reason).toContain("prediction");
  });

  it("also fails when the spoofer is free-running and unsynchronized", () => {
    const frame = surface(vlp16);
    const locked = attack({ frame });
    const free = attack({
      frame,
      spoofer: configureSpoofer("cao2023", { syncMode: "free", canPredict: true }),
    });
    expect(free.stats.removed * 5).toBeLessThan(locked.stats.removed);
  });
});

describe("range replacement above the MOT", () => {
  it("moves the genuine return instead of deleting it", () => {
    // Pre-MOT-filter, the pcaps show removal as genuine echoes being
    // *range-replaced* on their own ray - target at 2.01 m reappearing at ~11 m.
    const frame = surface(vlp16, 60);
    const res = attack({ frame, attackerRangeM: 11 });
    expect(res.stats.outcome).toBe(OUTCOMES.REPLACED);
    expect(res.stats.replaced).toBeGreaterThan(0);
    expect(res.replacement.r.length).toBe(res.stats.replaced);
    for (let i = 0; i < res.replacement.r.length; i++) {
      expect(res.replacement.r[i]).toBeCloseTo(11, 0.3);
    }
  });

  it("saturates the intensity of replacement points, as the real spoofer does", () => {
    const frame = surface(vlp16, 60);
    const res = attack({ frame, attackerRangeM: 11 });
    for (let i = 0; i < res.replacement.intensity.length; i++) {
      expect(res.replacement.intensity[i]).toBe(255);
    }
  });

  it("does nothing when the attacker echo is farther than the genuine return", () => {
    const frame = surface(vlp16, 60);
    const res = attack({ frame, attackerRangeM: 80 });
    expect(res.stats.removed).toBe(0);
    expect(res.stats.replaced).toBe(0);
    expect(res.stats.outcome).toBe(OUTCOMES.OCCLUDED);
  });
});

describe("removal accounting", () => {
  it("reports the rate relative to genuine returns actually inside the envelope", () => {
    const frame = surface(vlp16);
    const res = attack({ frame });
    expect(res.stats.envelopeReturns).toBeGreaterThan(0);
    expect(res.stats.envelopeMeanRate).toBeCloseTo(
      res.stats.removed / res.stats.envelopeReturns,
      1e-9
    );
    expect(res.stats.envelopeMeanRate).toBeLessThanOrEqual(1);
  });

  it("is reproducible for a fixed seed", () => {
    const frame = surface(vlp16);
    expect(attack({ frame, seed: 21 }).stats.removed).toBe(
      attack({ frame, seed: 21 }).stats.removed
    );
  });

  it("returns an empty result for an empty envelope", () => {
    const frame = surface(vlp16);
    const res = attack({ frame, positionXYZ: [500, 0, 0] });
    expect(res.stats.removed).toBe(0);
    expect(res.stats.outcome).toBe(OUTCOMES.INFEASIBLE);
  });
});
