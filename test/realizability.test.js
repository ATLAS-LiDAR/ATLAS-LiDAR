import { describe, it, expect } from "./harness.js";
import { configureSensor } from "../src/core/sensor.js";
import { configureSpoofer, reachableCells } from "../src/core/spoofer.js";
import { emptyFrame } from "./fixtures.js";
import {
  cellsAndFill,
  analyzeAttack,
  classifyRealizability,
  firingBudget,
  DEMONSTRATED_ENVELOPE,
  VERDICTS,
} from "../src/core/realizability.js";

const sensor = configureSensor("hdl32e");
const vlp16 = configureSensor("vlp16");

describe("cellsAndFill", () => {
  it("computes the sampled cell count and occupancy fraction", () => {
    const r = cellsAndFill({ azSpanDeg: 10, nBeamsSpanned: 32, nPoints: 160, azResDeg: 0.3316 });
    expect(r.nAz).toBeCloseTo(10 / 0.3316, 1e-6);
    expect(r.cells).toBeCloseTo((10 / 0.3316) * 32, 1e-6);
    expect(r.fill).toBeCloseTo(160 / r.cells, 1e-9);
  });

  it("floors the azimuth cell count at one, as the reference implementation does", () => {
    const r = cellsAndFill({ azSpanDeg: 0.01, nBeamsSpanned: 1, nPoints: 1, azResDeg: 0.3316 });
    expect(r.nAz).toBe(1);
  });
});

describe("DEMONSTRATED_ENVELOPE", () => {
  it("records the measured envelope from the real VLP-16 capture", () => {
    const e = DEMONSTRATED_ENVELOPE;
    expect(e.azSpanDeg).toBeGreaterThan(6.5);
    expect(e.azSpanDeg).toBeLessThan(8);
    expect(e.beamsSpanned).toBe(16);
    expect(e.pointsPerRevolution).toBeGreaterThan(500);
    expect(e.cellFillUnique).toBeGreaterThan(0.2);
    expect(e.cellFillUnique).toBeLessThan(0.4);
  });

  it("keeps the as-published figures alongside the corrected ones", () => {
    // osf_k6nf2/realizability.py assumes 0.40 deg azimuth resolution for a 10 Hz
    // VLP-16, but the datasheet gives 0.192 deg at that spin rate (300k pts/s
    // over 16 channels). The coarse value halves the cell count and so doubles
    // measured fill - and pushes the dual-return figure to 2.219 returns/cell,
    // above the hard limit of 2. Both are recorded so the comparison is explicit.
    expect(DEMONSTRATED_ENVELOPE.azResDeg).toBeCloseTo(0.192, 0.01);
    expect(DEMONSTRATED_ENVELOPE.asPublished.azResDeg).toBeCloseTo(0.40, 1e-9);
    expect(DEMONSTRATED_ENVELOPE.asPublished.cellFillUnique).toBeGreaterThan(
      DEMONSTRATED_ENVELOPE.cellFillUnique * 1.9
    );
    expect(DEMONSTRATED_ENVELOPE.asPublished.cellFillDual).toBeGreaterThan(2.0);
  });
});

describe("classifyRealizability", () => {
  const base = { azSpanDeg: 7, beamsSpanned: 16, nPoints: 300, fill: 0.28, fillUnique: 0.28 };

  it("calls more than one return per cell physically impossible", () => {
    const v = classifyRealizability({ ...base, fill: 1.4 }, { maxReturnsPerCell: 1 });
    expect(v.verdict).toBe(VERDICTS.IMPOSSIBLE);
    expect(v.reasons.join(" ")).toContain("per cell");
  });

  it("allows two returns per cell on a dual-return sensor", () => {
    const v = classifyRealizability({ ...base, fill: 1.4 }, { maxReturnsPerCell: 2 });
    expect(v.verdict).not.toBe(VERDICTS.IMPOSSIBLE);
  });

  it("calls an attack inside the measured envelope demonstrated", () => {
    const v = classifyRealizability(base, { maxReturnsPerCell: 1 });
    expect(v.verdict).toBe(VERDICTS.DEMONSTRATED);
  });

  it("calls a wider but sub-45-degree attack within stated capability", () => {
    const v = classifyRealizability({ ...base, azSpanDeg: 20 }, { maxReturnsPerCell: 1 });
    expect(v.verdict).toBe(VERDICTS.WITHIN_CAPABILITY);
  });

  it("flags spans beyond the 45 degree capability as extrapolation", () => {
    const v = classifyRealizability({ ...base, azSpanDeg: 60 }, { maxReturnsPerCell: 1 });
    expect(v.verdict).toBe(VERDICTS.EXTRAPOLATION);
    expect(v.reasons.join(" ")).toContain("45");
  });

  it("flags a point budget beyond what was demonstrated", () => {
    const v = classifyRealizability({ ...base, nPoints: 5000 }, { maxReturnsPerCell: 1 });
    expect(v.verdict).not.toBe(VERDICTS.DEMONSTRATED);
  });
});

describe("analyzeAttack", () => {
  it("measures span, beams and fill from an actual reachable set", () => {
    const spoofer = configureSpoofer("cao2023");
    const reach = reachableCells({
      frame: emptyFrame(vlp16),
      sensor: vlp16,
      spoofer,
      positionXYZ: [40, 0, 0],
    });
    const a = analyzeAttack({ reach, sensor: vlp16, nInjected: reach.cells.length });
    expect(a.azSpanDeg).toBeGreaterThan(5);
    expect(a.azSpanDeg).toBeLessThan(9);
    expect(a.beamsSpanned).toBe(16);
    expect(a.fillUnique).toBeLessThanOrEqual(1.0);
  });

  it("never reports fill above one for an attack that respects one point per cell", () => {
    // The injection model enforces a single spoofed return per cell, so its
    // output is realizable by construction; this guards that invariant.
    const spoofer = configureSpoofer("cao2023", { powerRatio: 50 });
    const reach = reachableCells({
      frame: emptyFrame(sensor),
      sensor,
      spoofer,
      positionXYZ: [40, 0, 0],
    });
    const a = analyzeAttack({ reach, sensor, nInjected: reach.cells.length });
    expect(a.fill).toBeLessThanOrEqual(1.0);
    expect(classifyRealizability(a, { maxReturnsPerCell: 1 }).verdict).not.toBe(
      VERDICTS.IMPOSSIBLE
    );
  });

  it("returns a zero analysis for an attack with no reachable cells", () => {
    const reach = reachableCells({
      frame: emptyFrame(sensor),
      sensor,
      spoofer: configureSpoofer("cao2023"),
      positionXYZ: [500, 0, 0],
    });
    const a = analyzeAttack({ reach, sensor, nInjected: 0 });
    expect(a.azSpanDeg).toBe(0);
    expect(a.fill).toBe(0);
  });
});

describe("firingBudget", () => {
  it("limits points per revolution by PRF over spin rate", () => {
    const b = firingBudget({ spoofer: configureSpoofer("cao2023", { prfHz: 20000 }), sensor });
    expect(b.maxCellsPerRevolution).toBe(2000);
  });

  it("reports whether a required cell count fits the budget", () => {
    const b = firingBudget({
      spoofer: configureSpoofer("cao2023", { prfHz: 20000 }),
      sensor,
      requiredCells: 500,
    });
    expect(b.withinBudget).toBeTruthy();
    const c = firingBudget({
      spoofer: configureSpoofer("cao2023", { prfHz: 2000 }),
      sensor,
      requiredCells: 5000,
    });
    expect(c.withinBudget).toBeFalsy();
    expect(c.shortfall).toBe(5000 - 200);
  });

  it("scales with the victim's spin rate", () => {
    const slow = firingBudget({ spoofer: configureSpoofer("cao2023"), sensor: configureSensor("hdl32e", { hz: 5 }) });
    const fast = firingBudget({ spoofer: configureSpoofer("cao2023"), sensor: configureSensor("hdl32e", { hz: 20 }) });
    expect(slow.maxCellsPerRevolution).toBeGreaterThan(fast.maxCellsPerRevolution);
  });
});
