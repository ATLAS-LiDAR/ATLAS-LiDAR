import { describe, it, expect } from "./harness.js";
import { rng, uniform } from "./util.js";
import { configureSensor } from "../src/core/sensor.js";
import { PointFrame } from "../src/core/frame.js";
import { frameOf, atCell, emptyFrame } from "./fixtures.js";
import {
  sectorDwellTime,
  pulsesDuringDwell,
  victimCellsInSector,
  sweptPointCount,
  advanceSweep,
  sweepAzToCell,
  beamInSector,
  sweepRevolutions,
} from "../src/core/sweep.js";

const DEG = Math.PI / 180;
const sensor = configureSensor("hdl64e");

function randomFrame(n = 20000, seed = 5) {
  const next = rng(seed);
  const buf = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    buf[i * 4] = uniform(next, -60, 60);
    buf[i * 4 + 1] = uniform(next, -60, 60);
    buf[i * 4 + 2] = uniform(next, -3, 6);
    buf[i * 4 + 3] = uniform(next, 0, 1);
  }
  return PointFrame.fromXYZI(buf, sensor);
}

describe("sweptPointCount", () => {
  it("reveals nothing at the start of a revolution and everything at the end", () => {
    const frame = randomFrame();
    expect(sweptPointCount(frame, 0)).toBe(0);
    expect(sweptPointCount(frame, sensor.azCellCount)).toBe(frame.count);
  });

  it("is monotonically non-decreasing across the revolution", () => {
    const frame = randomFrame();
    let prev = 0;
    for (let a = 0; a <= sensor.azCellCount; a += 17) {
      const c = sweptPointCount(frame, a);
      expect(c).toBeGreaterThanOrEqual(prev);
      prev = c;
    }
  });

  it("only reveals points the beam has actually passed", () => {
    // The load-bearing claim: frame.order is grouped by cellKey =
    // azIdx * nBeams + beamIdx, which is monotonic in azIdx, so the CSR index is
    // ALREADY azimuth-sorted and the reveal is an O(1) offset lookup rather than
    // a per-frame scan. If that were false the animation would show points out
    // of order, so assert it directly.
    const frame = randomFrame();
    for (const cut of [1, 50, 400, 1200, 2000]) {
      const k = sweptPointCount(frame, cut);
      for (let i = 0; i < k; i++) {
        expect(frame.azIdx[frame.order[i]]).toBeLessThan(cut);
      }
      // and the very next point is at or beyond the cutoff
      if (k < frame.count) {
        expect(frame.azIdx[frame.order[k]]).toBeGreaterThanOrEqual(cut);
      }
    }
  });

  it("clamps out-of-range cell indices instead of reading past the offsets", () => {
    const frame = randomFrame(500, 9);
    expect(sweptPointCount(frame, -5)).toBe(0);
    expect(sweptPointCount(frame, sensor.azCellCount + 1000)).toBe(frame.count);
  });

  it("handles an empty frame", () => {
    const frame = emptyFrame(sensor);
    expect(sweptPointCount(frame, 0)).toBe(0);
    expect(sweptPointCount(frame, sensor.azCellCount)).toBe(0);
  });

  it("returns exactly the points of the columns already passed", () => {
    const frame = frameOf(sensor, [
      atCell(sensor, 10, 3, 20),
      atCell(sensor, 10, 4, 21),
      atCell(sensor, 50, 3, 22),
      atCell(sensor, 900, 3, 23),
    ]);
    expect(sweptPointCount(frame, 10)).toBe(0);
    expect(sweptPointCount(frame, 11)).toBe(2);
    expect(sweptPointCount(frame, 51)).toBe(3);
    expect(sweptPointCount(frame, 901)).toBe(4);
  });

  it("is fast enough to call every animation frame on a full cloud", () => {
    const frame = randomFrame(120000, 11);
    const t0 = performance.now();
    for (let i = 0; i < 2000; i++) sweptPointCount(frame, i);
    expect(performance.now() - t0).toBeLessThan(50);
  });
});

describe("advanceSweep", () => {
  it("advances one full revolution per 1/hz seconds at unit speed", () => {
    const az = advanceSweep(0, 1 / sensor.hz, sensor, 1);
    expect(az).toBeCloseTo(0, 1e-9); // wrapped back to the start
    const half = advanceSweep(0, 1 / sensor.hz / 2, sensor, 1);
    expect(half).toBeCloseTo(Math.PI, 1e-9);
  });

  it("scales with the speed multiplier", () => {
    const slow = advanceSweep(0, 0.01, sensor, 0.25);
    const fast = advanceSweep(0, 0.01, sensor, 1.0);
    expect(fast).toBeCloseTo(slow * 4, 1e-9);
  });

  it("always returns an azimuth in [0, 2pi)", () => {
    let az = 0;
    for (let i = 0; i < 500; i++) {
      az = advanceSweep(az, 0.037, sensor, 1.7);
      expect(az).toBeGreaterThanOrEqual(0);
      expect(az).toBeLessThan(2 * Math.PI);
    }
  });

  it("does not move when paused (dt or speed zero)", () => {
    expect(advanceSweep(1.2, 0, sensor, 1)).toBeCloseTo(1.2, 1e-12);
    expect(advanceSweep(1.2, 0.5, sensor, 0)).toBeCloseTo(1.2, 1e-12);
  });

  it("counts completed revolutions for a given elapsed time", () => {
    expect(sweepRevolutions(1.0, sensor, 1)).toBeCloseTo(sensor.hz, 1e-9);
  });
});

describe("sweepAzToCell", () => {
  it("maps azimuth onto the sensor's own column index", () => {
    expect(sweepAzToCell(0, sensor)).toBe(0);
    expect(sweepAzToCell(sensor.azRes * 3.5, sensor)).toBe(3);
    expect(sweepAzToCell(2 * Math.PI - 1e-12, sensor)).toBeLessThan(sensor.azCellCount);
  });
});

describe("beamInSector", () => {
  it("is true while the beam is inside the attacker's sector", () => {
    expect(beamInSector(0, 0, 20 * DEG)).toBeTruthy();
    expect(beamInSector(9 * DEG, 0, 20 * DEG)).toBeTruthy();
    expect(beamInSector(11 * DEG, 0, 20 * DEG)).toBeFalsy();
  });

  it("handles a sector spanning the wrap point", () => {
    // The attacker's window does not care where the encoder's zero happens to be.
    expect(beamInSector(0.01, 2 * Math.PI - 0.02, 10 * DEG)).toBeTruthy();
    expect(beamInSector(2 * Math.PI - 0.01, 0.02, 10 * DEG)).toBeTruthy();
  });

  it("is what makes the synchronization model visible", () => {
    // The attacker can only act while the victim's beam is in its window, so the
    // fraction of a revolution for which this is true is the duty cycle the
    // animation should show.
    const widthRad = 20 * DEG;
    let inside = 0;
    const N = 3600;
    for (let i = 0; i < N; i++) {
      if (beamInSector((i / N) * 2 * Math.PI, 0, widthRad)) inside++;
    }
    expect(inside / N).toBeCloseTo(widthRad / (2 * Math.PI), 0.01);
  });
});

describe("attacker firing vs the victim's scan", () => {
  it("computes how long the beam spends crossing a sector", () => {
    // a 20 deg sector at 10 Hz: (20/360) * 100 ms = 5.56 ms
    const dwell = sectorDwellTime(sensor, (20 * Math.PI) / 180);
    expect(dwell).toBeCloseTo((20 / 360) * (1 / sensor.hz), 1e-9);
    expect(dwell * 1000).toBeCloseTo(5.56, 0.05);
  });

  it("scales dwell with sector width and inversely with spin rate", () => {
    const wide = sectorDwellTime(sensor, (40 * Math.PI) / 180);
    const narrow = sectorDwellTime(sensor, (20 * Math.PI) / 180);
    expect(wide).toBeCloseTo(narrow * 2, 1e-9);
    const fast = sectorDwellTime(configureSensor("hdl64e", { hz: 20 }), (20 * Math.PI) / 180);
    expect(fast).toBeCloseTo(narrow / 2, 1e-9);
  });

  it("counts the attacker pulses that land during one crossing", () => {
    const dwell = sectorDwellTime(sensor, (20 * Math.PI) / 180);
    expect(pulsesDuringDwell(100000, dwell)).toBe(Math.floor(100000 * dwell));
    expect(pulsesDuringDwell(0, dwell)).toBe(0);
  });

  it("counts the victim cells inside the sector for comparison", () => {
    // The interesting comparison: the attacker gets far fewer pulses than there
    // are cells to fill, which is the firing-budget constraint made concrete.
    const cells = victimCellsInSector(sensor, (20 * Math.PI) / 180, (16 * Math.PI) / 180);
    expect(cells).toBeGreaterThan(1000);
    const dwell = sectorDwellTime(sensor, (20 * Math.PI) / 180);
    expect(pulsesDuringDwell(100000, dwell)).toBeLessThan(cells);
  });

  it("counts only the beams the sector actually spans vertically", () => {
    const narrowEl = victimCellsInSector(sensor, (20 * Math.PI) / 180, (2 * Math.PI) / 180);
    const wideEl = victimCellsInSector(sensor, (20 * Math.PI) / 180, (30 * Math.PI) / 180);
    expect(wideEl).toBeGreaterThan(narrowEl);
  });
});
