import { describe, it, expect } from "./harness.js";
import { configureSensor, C_LIGHT } from "../src/core/sensor.js";
import {
  apparentRange,
  delayForRange,
  minRelayRange,
  requiresPrediction,
  rangeErrorStd,
  gateDutyCycle,
  syncHitProbability,
  tofBudget,
  predictedFireTime,
} from "../src/core/tof.js";

const HALF_C = C_LIGHT / 2; // 149896229 m/s

describe("apparent range from time of flight", () => {
  it("puts a zero-latency relay spoofer exactly at its own standoff distance", () => {
    // The load-bearing physical fact: a reactive spoofer must wait for the
    // victim's pulse to reach it (d/c) and its reply must travel back (d/c), so
    // the victim measures ToF = 2d/c and reports range = d. A phantom cannot
    // appear closer than the spoofer without predicting the next pulse.
    expect(apparentRange({ standoffM: 40, latencyS: 0, delayS: 0 })).toBeCloseTo(40, 1e-9);
    expect(apparentRange({ standoffM: 7.5, latencyS: 0, delayS: 0 })).toBeCloseTo(7.5, 1e-9);
  });

  it("converts electronic latency into added apparent range at c/2", () => {
    expect(apparentRange({ standoffM: 40, latencyS: 10e-9, delayS: 0 })).toBeCloseTo(
      40 + 10e-9 * HALF_C,
      1e-9
    );
    // 10 ns of latency is 1.5 m of range error
    expect(10e-9 * HALF_C).toBeCloseTo(1.49896229, 1e-8);
  });

  it("adds programmed delay on top of latency", () => {
    const r = apparentRange({ standoffM: 12, latencyS: 5e-9, delayS: 200e-9 });
    expect(r).toBeCloseTo(12 + 205e-9 * HALF_C, 1e-9);
  });
});

describe("delayForRange", () => {
  it("inverts apparentRange", () => {
    const d = delayForRange({ targetRangeM: 55, standoffM: 20, latencyS: 8e-9 });
    expect(apparentRange({ standoffM: 20, latencyS: 8e-9, delayS: d })).toBeCloseTo(55, 1e-9);
  });

  it("returns a negative delay when the target is nearer than the spoofer", () => {
    // A negative delay is not a knob you can turn - it means "fire before the
    // victim's pulse arrives", which requires pulse prediction.
    const d = delayForRange({ targetRangeM: 10, standoffM: 40, latencyS: 0 });
    expect(d).toBeLessThan(0);
    expect(d).toBeCloseTo((2 * (10 - 40)) / C_LIGHT, 1e-15);
  });
});

describe("minRelayRange and requiresPrediction", () => {
  it("floors a reactive spoofer at standoff plus latency range", () => {
    expect(minRelayRange({ standoffM: 40, latencyS: 0 })).toBeCloseTo(40, 1e-9);
    expect(minRelayRange({ standoffM: 40, latencyS: 20e-9 })).toBeCloseTo(
      40 + 20e-9 * HALF_C,
      1e-9
    );
  });

  it("flags near-field phantoms as needing prediction", () => {
    const cfg = { standoffM: 40, latencyS: 10e-9 };
    expect(requiresPrediction(5, cfg)).toBeTruthy();
    expect(requiresPrediction(39, cfg)).toBeTruthy();
    expect(requiresPrediction(60, cfg)).toBeFalsy();
    // exactly at the floor is reachable
    expect(requiresPrediction(minRelayRange(cfg) + 1e-6, cfg)).toBeFalsy();
  });
});

describe("timing jitter maps to range error", () => {
  it("converts a timing standard deviation to a range standard deviation", () => {
    expect(rangeErrorStd(1e-9)).toBeCloseTo(0.149896229, 1e-9);
    expect(rangeErrorStd(0)).toBeCloseTo(0, 1e-12);
  });

  it("is linear, so 10x the jitter is 10x the range spread", () => {
    expect(rangeErrorStd(10e-9)).toBeCloseTo(10 * rangeErrorStd(1e-9), 1e-9);
  });
});

describe("synchronization", () => {
  const vlp16 = configureSensor("vlp16");

  it("computes the free-running duty cycle from gate width over firing period", () => {
    // An unsynchronized spoofer is only heard if its pulse happens to fall in
    // the receiver's open gate. That is a ~2% chance on a VLP-16, which is why
    // every practical attack in the literature synchronizes.
    const duty = gateDutyCycle(vlp16);
    expect(duty).toBeCloseTo(vlp16.rangeGate / vlp16.firingPeriod, 1e-12);
    expect(duty).toBeGreaterThan(0.005);
    expect(duty).toBeLessThan(0.05);
  });

  it("free-running mode ignores jitter and returns the duty cycle", () => {
    const p = syncHitProbability({ sensor: vlp16, mode: "free", jitterS: 1e-9 });
    expect(p).toBeCloseTo(gateDutyCycle(vlp16), 1e-12);
  });

  it("a well-locked spoofer lands essentially every pulse", () => {
    const p = syncHitProbability({
      sensor: vlp16,
      mode: "locked",
      jitterS: 1e-12,
      toleranceM: 0.5,
    });
    expect(p).toBeGreaterThan(0.999);
  });

  it("degrades monotonically as lock jitter grows", () => {
    const p = (j) =>
      syncHitProbability({ sensor: vlp16, mode: "locked", jitterS: j, toleranceM: 0.5 });
    const a = p(1e-12);
    const b = p(2e-9);
    const c = p(50e-9);
    expect(a).toBeGreaterThan(b);
    expect(b).toBeGreaterThan(c);
    expect(c).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThanOrEqual(1);
  });

  it("never exceeds the free-running floor when lock is meaningless", () => {
    // With enormous jitter a "locked" spoofer is no better than free-running.
    const p = syncHitProbability({
      sensor: vlp16,
      mode: "locked",
      jitterS: 1e-3,
      toleranceM: 0.5,
    });
    expect(p).toBeGreaterThanOrEqual(gateDutyCycle(vlp16) * 0.5);
    expect(p).toBeLessThan(0.1);
  });
});

describe("tofBudget", () => {
  const hdl32 = configureSensor("hdl32e");

  it("reports the reachable range interval for a reactive spoofer", () => {
    const b = tofBudget({ sensor: hdl32, standoffM: 25, latencyS: 10e-9, predictive: false });
    expect(b.minRange).toBeCloseTo(25 + 10e-9 * HALF_C, 1e-9);
    expect(b.maxRange).toBeCloseTo(hdl32.maxRange, 1e-9);
    expect(b.requiresPrediction).toBeFalsy();
  });

  it("opens up the near field only when prediction is enabled", () => {
    const reactive = tofBudget({ sensor: hdl32, standoffM: 25, latencyS: 0, predictive: false });
    const predictive = tofBudget({ sensor: hdl32, standoffM: 25, latencyS: 0, predictive: true });
    expect(reactive.minRange).toBeCloseTo(25, 1e-9);
    expect(predictive.minRange).toBeCloseTo(hdl32.minRange, 1e-9);
    expect(predictive.minRange).toBeLessThan(reactive.minRange);
  });

  it("caps max range at the receiver's gate, not just the spec sheet", () => {
    const narrow = configureSensor("hdl32e", { maxRange: 100 });
    const b = tofBudget({ sensor: narrow, standoffM: 5, latencyS: 0, predictive: false });
    expect(b.maxRange).toBeLessThanOrEqual(narrow.tofToRange(narrow.rangeGate));
    expect(b.maxDelayS).toBeGreaterThan(0);
    expect(
      apparentRange({ standoffM: 5, latencyS: 0, delayS: b.maxDelayS })
    ).toBeCloseTo(b.maxRange, 1e-6);
  });

  it("reports an empty budget when the spoofer stands beyond the victim's range", () => {
    const b = tofBudget({ sensor: hdl32, standoffM: 150, latencyS: 0, predictive: false });
    expect(b.feasible).toBeFalsy();
  });
});

describe("predictedFireTime", () => {
  const hdl32 = configureSensor("hdl32e");

  it("returns the time within the revolution at which a cell is sampled", () => {
    expect(predictedFireTime(hdl32, 0, 0)).toBeCloseTo(0, 1e-12);
    expect(predictedFireTime(hdl32, 100, 0)).toBeCloseTo(hdl32.timeOfAzCell(100), 1e-12);
  });

  it("shifts with the victim's azimuth phase", () => {
    const t = predictedFireTime(hdl32, 100, 0.25 * hdl32.revolutionPeriod);
    expect(t).toBeCloseTo(hdl32.timeOfAzCell(100) + 0.25 * hdl32.revolutionPeriod, 1e-12);
  });

  it("wraps within one revolution", () => {
    const t = predictedFireTime(hdl32, hdl32.azCellCount - 1, 0.9 * hdl32.revolutionPeriod);
    expect(t).toBeGreaterThanOrEqual(0);
    expect(t).toBeLessThan(hdl32.revolutionPeriod);
  });
});
