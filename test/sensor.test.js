import { describe, it, expect } from "./harness.js";
import {
  SENSORS,
  getSensor,
  configureSensor,
  C_LIGHT,
  listSensors,
} from "../src/core/sensor.js";

const DEG = Math.PI / 180;

describe("sensor registry", () => {
  it("exposes every sensor with a complete, self-consistent spec", () => {
    const ids = listSensors();
    expect(ids.length).toBeGreaterThan(3);
    for (const id of ids) {
      const s = getSensor(id);
      expect(typeof s.name).toBe("string");
      expect(s.nBeams).toBeGreaterThan(0);
      expect(s.inclinations).toHaveLength(s.nBeams);
      expect(s.maxRange).toBeGreaterThan(0);
      expect(s.hz).toBeGreaterThan(0);
      // minRange is the Minimum Operational Threshold the firmware enforces;
      // removal attacks depend on it existing.
      expect(s.minRange).toBeGreaterThan(0);
      expect(["strongest", "last", "dual"]).toContain(s.returnMode);
    }
  });

  it("keeps inclinations strictly ascending so binary search is valid", () => {
    for (const id of listSensors()) {
      const inc = getSensor(id).inclinations;
      for (let i = 1; i < inc.length; i++) {
        expect(inc[i]).toBeGreaterThan(inc[i - 1]);
      }
    }
  });

  it("throws a useful error on an unknown sensor id", () => {
    expect(() => getSensor("nope")).toThrow("nope");
  });
});

describe("azimuth lattice", () => {
  it("derives cell count from the measured nuScenes azimuth resolution", () => {
    const s = getSensor("hdl32e");
    expect(s.azRes / DEG).toBeCloseTo(0.33159202, 1e-7);
    expect(s.azCellCount).toBe(Math.ceil((2 * Math.PI) / s.azRes));
    expect(s.azCellCount).toBe(1086);
  });

  it("derives cell count for KITTI's HDL-64E", () => {
    const s = getSensor("hdl64e");
    expect(s.azRes / DEG).toBeCloseTo(0.1728, 1e-7);
    expect(s.azCellCount).toBe(2084);
  });

  it("reports points per revolution as the full sampled lattice size", () => {
    const s = getSensor("hdl64e");
    expect(s.pointsPerRevolution).toBe(s.nBeams * s.azCellCount);
  });
});

describe("derived point rate cross-checks the datasheet", () => {
  // This is the strongest available check that the measured assets are right:
  // nBeams * azCellCount * hz must land on the manufacturer's quoted rate.
  it("HDL-32E derives ~350k points/sec (datasheet 350k single-return)", () => {
    const s = getSensor("hdl32e");
    expect(s.pointRate).toBeGreaterThan(330000);
    expect(s.pointRate).toBeLessThan(365000);
  });

  it("HDL-64E derives ~1.33M points/sec (datasheet 1.33M single-return)", () => {
    const s = getSensor("hdl64e");
    expect(s.pointRate).toBeGreaterThan(1290000);
    expect(s.pointRate).toBeLessThan(1370000);
  });

  it("VLP-16 at 10 Hz derives ~0.19 deg azimuth resolution from its 300k rate", () => {
    // Datasheet: 300k pts/s over 16 channels => 18.75 kHz per laser => at 10 Hz
    // that is 1875 azimuth columns, i.e. 0.192 deg -- NOT the 0.40 deg that
    // osf_k6nf2/realizability.py assumes for the same 10 Hz capture. See the
    // note in realizability.js; the discrepancy scales cell-fill by 2x.
    const s = getSensor("vlp16");
    expect(s.hz).toBe(10);
    expect(s.azRes / DEG).toBeCloseTo(0.192, 0.01);
    expect(s.azCellCount).toBeCloseTo(1875, 2);
  });

  it("VLP-16 at 20 Hz halves the azimuth resolution at constant point rate", () => {
    const s = configureSensor("vlp16", { hz: 20 });
    expect(s.pointRate).toBeCloseTo(getSensor("vlp16").pointRate, 1);
    expect(s.azCellCount).toBeCloseTo(937, 2);
    expect(s.azRes / DEG).toBeCloseTo(0.384, 0.01);
  });
});

describe("pulse timing", () => {
  it("gives each laser a PRF equal to columns per revolution times spin rate", () => {
    const s = getSensor("vlp16");
    expect(s.laserPRF).toBeCloseTo(s.azCellCount * s.hz, 1e-6);
    // ~18.75 kHz per laser, i.e. a 53.3 us firing cycle
    expect(s.firingPeriod).toBeCloseTo(1 / s.laserPRF, 1e-12);
    expect(s.firingPeriod).toBeGreaterThan(50e-6);
    expect(s.firingPeriod).toBeLessThan(56e-6);
  });

  it("maps azimuth cell index to a time offset within the revolution", () => {
    const s = getSensor("hdl32e");
    expect(s.timeOfAzCell(0)).toBeCloseTo(0, 1e-12);
    expect(s.timeOfAzCell(s.azCellCount)).toBeCloseTo(1 / s.hz, 1e-9);
    // strictly increasing: this ordering is what a synchronized attacker exploits
    expect(s.timeOfAzCell(5)).toBeGreaterThan(s.timeOfAzCell(4));
  });

  it("has a range gate wide enough for its own max range", () => {
    for (const id of listSensors()) {
      const s = getSensor(id);
      const maxToF = (2 * s.maxRange) / C_LIGHT;
      expect(s.rangeGate).toBeGreaterThanOrEqual(maxToF);
      // and the gate must close before the next pulse, or ranges alias
      expect(s.rangeGate).toBeLessThanOrEqual(s.firingPeriod);
    }
  });

  it("converts between range and time of flight", () => {
    const s = getSensor("vlp16");
    expect(s.rangeToToF(150)).toBeCloseTo((2 * 150) / C_LIGHT, 1e-15);
    expect(s.tofToRange(s.rangeToToF(37.5))).toBeCloseTo(37.5, 1e-9);
  });
});

describe("configureSensor", () => {
  it("returns a new spec without mutating the registry", () => {
    const before = getSensor("vlp16").hz;
    const modified = configureSensor("vlp16", { hz: 20 });
    expect(modified.hz).toBe(20);
    expect(getSensor("vlp16").hz).toBe(before);
    expect(SENSORS.vlp16.hz).toBe(before);
  });

  it("recomputes the derived lattice when return mode changes to dual", () => {
    const single = configureSensor("vlp16", { returnMode: "strongest" });
    const dual = configureSensor("vlp16", { returnMode: "dual" });
    expect(single.pTwo).toBe(0);
    // Dual return means a cell can carry two echoes; the real capture measured
    // 2.03 returns/ray in dual mode vs 1.02 in last mode.
    expect(dual.pTwo).toBeGreaterThan(0.9);
  });

  it("accepts an explicit azimuth resolution override and rederives point rate", () => {
    const s = configureSensor("vlp16", { azResDeg: 0.4 });
    expect(s.azRes / DEG).toBeCloseTo(0.4, 1e-9);
    expect(s.azCellCount).toBe(Math.ceil(360 / 0.4));
    expect(s.pointRate).toBeCloseTo(s.nBeams * s.azCellCount * s.hz, 1e-6);
  });
});
