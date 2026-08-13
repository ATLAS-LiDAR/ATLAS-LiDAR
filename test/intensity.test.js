import { describe, it, expect } from "./harness.js";
import { rng } from "./util.js";
import {
  bandForRange,
  sampleIntensity,
  assignIntensity,
  INTENSITY_MODES,
  realCarIntensityP99,
} from "../src/core/intensity.js";

describe("bandForRange", () => {
  it("selects the band containing the range, half-open like ATLAS", () => {
    expect(bandForRange("nuscenes", 5).rMax).toBe(10);
    expect(bandForRange("nuscenes", 10).rMin).toBe(10);
    expect(bandForRange("nuscenes", 25).rMin).toBe(20);
  });

  it("clamps beyond the last band rather than returning nothing", () => {
    // ATLAS's get_stats_for_range returns None past 80 m and the caller then
    // silently leaves intensity untouched; clamping is the honest behaviour.
    const b = bandForRange("nuscenes", 500);
    expect(b).toBeTruthy();
    expect(b.rMax).toBe(80);
  });

  it("has separate statistics per dataset with different scales", () => {
    // nuScenes intensity is 0-255, KITTI is normalised 0-1.
    expect(bandForRange("nuscenes", 15).p99).toBeGreaterThan(10);
    expect(bandForRange("kitti", 15).p99).toBeLessThanOrEqual(1);
  });
});

describe("sampleIntensity", () => {
  it("stays inside the measured [p1, p99] clip for its band", () => {
    const next = rng(3);
    const band = bandForRange("nuscenes", 15);
    for (let i = 0; i < 2000; i++) {
      const v = sampleIntensity({ dataset: "nuscenes", r: 15, next });
      expect(v).toBeGreaterThanOrEqual(band.p1);
      expect(v).toBeLessThanOrEqual(band.p99);
    }
  });

  it("centres near exp(logMu), the lognormal median", () => {
    const next = rng(9);
    const band = bandForRange("nuscenes", 15);
    const vals = [];
    for (let i = 0; i < 4000; i++) vals.push(sampleIntensity({ dataset: "nuscenes", r: 15, next }));
    vals.sort((a, b) => a - b);
    const median = vals[vals.length >> 1];
    const expected = Math.min(band.p99, Math.max(band.p1, Math.exp(band.logMu)));
    expect(median).toBeCloseTo(expected, expected * 0.25);
  });

  it("is reproducible for a given seed", () => {
    const a = sampleIntensity({ dataset: "nuscenes", r: 20, next: rng(42) });
    const b = sampleIntensity({ dataset: "nuscenes", r: 20, next: rng(42) });
    expect(a).toBe(b);
  });
});

describe("assignIntensity modes", () => {
  const ranges = new Float64Array([5, 15, 25, 45, 70]);

  it("range-conditioned keeps phantom points in-distribution", () => {
    const out = assignIntensity({
      mode: "range-conditioned",
      dataset: "nuscenes",
      ranges,
      next: rng(1),
    });
    expect(out).toHaveLength(ranges.length);
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).toBeGreaterThan(0);
      expect(out[i]).toBeLessThanOrEqual(bandForRange("nuscenes", ranges[i]).p99);
    }
  });

  it("saturated reproduces the real spoofer: every return at full scale", () => {
    // The real capture measures median 255/255 with fraction==255 of 1.000.
    const out = assignIntensity({
      mode: "saturated",
      dataset: "nuscenes",
      ranges,
      next: rng(1),
      scaleMax: 255,
    });
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(255);
  });

  it("saturated intensity is far outside the real-return distribution", () => {
    // This is why closing the intensity gap collapses ASR by 85-95%: it is an
    // out-of-distribution effect in the top 1% tail, not a brightness trend.
    const sat = assignIntensity({
      mode: "saturated",
      dataset: "nuscenes",
      ranges,
      next: rng(1),
      scaleMax: 255,
    });
    expect(sat[0]).toBeGreaterThan(realCarIntensityP99("nuscenes") * 2);
  });

  it("constant mode forces a chosen value, for dose-response sweeps", () => {
    const out = assignIntensity({
      mode: "constant",
      dataset: "nuscenes",
      ranges,
      next: rng(1),
      constantValue: 100,
    });
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(100);
  });

  it("exposes the mode list the UI offers", () => {
    expect(INTENSITY_MODES).toContain("range-conditioned");
    expect(INTENSITY_MODES).toContain("saturated");
    expect(INTENSITY_MODES).toContain("constant");
  });

  it("rejects an unknown mode instead of silently defaulting", () => {
    expect(() =>
      assignIntensity({ mode: "wat", dataset: "nuscenes", ranges, next: rng(1) })
    ).toThrow("wat");
  });
});
