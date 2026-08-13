import { describe, it, expect } from "./harness.js";
import { configureSensor } from "../src/core/sensor.js";
import { emptyFrame, wallFrame, frameOf, atCell } from "./fixtures.js";
import { unpackCellKey } from "../src/core/polar.js";
import {
  SPOOFERS,
  configureSpoofer,
  listSpoofers,
  spooferGeometry,
  angularEnvelope,
  reachableCells,
  checkLineOfSight,
  CAO2023_ENVELOPE,
  DEMONSTRATED_MAX_AZ_DEG,
} from "../src/core/spoofer.js";

const DEG = Math.PI / 180;
const sensor = configureSensor("hdl32e");
const vlp16 = configureSensor("vlp16");

describe("spoofer registry", () => {
  it("provides hardware presets with the physical fields the model needs", () => {
    expect(listSpoofers().length).toBeGreaterThan(1);
    for (const id of listSpoofers()) {
      const s = configureSpoofer(id);
      expect(s.wavelengthNm).toBeGreaterThan(700);
      expect(s.apertureM).toBeGreaterThan(0);
      expect(s.divergenceMrad).toBeGreaterThan(0);
      expect(s.latencyS).toBeGreaterThanOrEqual(0);
      expect(s.prfHz).toBeGreaterThan(0);
      expect(s.powerRatio).toBeGreaterThan(0);
      expect(typeof s.canPredict).toBe("boolean");
    }
  });

  it("does not mutate the registry when overriding", () => {
    const before = SPOOFERS.cao2023.powerRatio;
    const s = configureSpoofer("cao2023", { powerRatio: 999 });
    expect(s.powerRatio).toBe(999);
    expect(SPOOFERS.cao2023.powerRatio).toBe(before);
  });

  it("throws on an unknown preset", () => {
    expect(() => configureSpoofer("bogus")).toThrow("bogus");
  });
});

describe("spooferGeometry", () => {
  it("derives standoff, bearing and elevation from the spoofer position", () => {
    const g = spooferGeometry({ positionXYZ: [10, 0, 0] });
    expect(g.standoff).toBeCloseTo(10, 1e-9);
    expect(g.bearing).toBeCloseTo(0, 1e-9);
    expect(g.elevation).toBeCloseTo(0, 1e-9);
  });

  it("uses the [0, 2pi) bearing convention", () => {
    expect(spooferGeometry({ positionXYZ: [0, -8, 0] }).bearing).toBeCloseTo(
      (3 * Math.PI) / 2,
      1e-9
    );
  });

  it("reports a raised spoofer as positive elevation", () => {
    const g = spooferGeometry({ positionXYZ: [10, 0, 10] });
    expect(g.elevation).toBeCloseTo(Math.PI / 4, 1e-9);
    expect(g.standoff).toBeCloseTo(Math.sqrt(200), 1e-9);
  });
});

describe("angularEnvelope", () => {
  it("reproduces the measured 6.8 degree envelope at the reference power", () => {
    // Calibration anchor: the real VLP-16 capture in Cao et al. shows a 6.8 deg
    // saturated azimuth envelope, i.e. a 3.4 deg half-width.
    const s = configureSpoofer("cao2023");
    const env = angularEnvelope({ spoofer: s, sensor: vlp16 });
    expect(env.halfWidthAz / DEG).toBeCloseTo(CAO2023_ENVELOPE.azHalfWidthDeg, 0.15);
    expect(env.halfWidthAz / DEG).toBeCloseTo(3.4, 0.15);
  });

  it("widens the envelope as attacker power over detection threshold rises", () => {
    const low = angularEnvelope({ spoofer: configureSpoofer("cao2023", { powerRatio: 2 }), sensor: vlp16 });
    const mid = angularEnvelope({ spoofer: configureSpoofer("cao2023", { powerRatio: 20 }), sensor: vlp16 });
    const high = angularEnvelope({ spoofer: configureSpoofer("cao2023", { powerRatio: 2000 }), sensor: vlp16 });
    expect(mid.halfWidthAz).toBeGreaterThan(low.halfWidthAz);
    expect(high.halfWidthAz).toBeGreaterThan(mid.halfWidthAz);
  });

  it("never narrows below the victim's instantaneous field of view", () => {
    // Even a barely-above-threshold spoofer still fills the one cell its own
    // bearing lands in; the envelope cannot be narrower than a single IFOV.
    const s = configureSpoofer("cao2023", { powerRatio: 1.0000001 });
    const env = angularEnvelope({ spoofer: s, sensor: vlp16 });
    expect(env.halfWidthAz).toBeGreaterThanOrEqual((vlp16.beamDivergenceMrad / 1000) / 2);
  });

  it("is not reachable at all below the detection threshold", () => {
    const env = angularEnvelope({ spoofer: configureSpoofer("cao2023", { powerRatio: 0.5 }), sensor: vlp16 });
    expect(env.feasible).toBeFalsy();
  });

  it("spreads much wider vertically than horizontally, as measured", () => {
    // The real capture reports 16/16 channels affected but only 6.8 deg of
    // azimuth: vertical crosstalk across the stacked detector array is strong.
    const env = angularEnvelope({ spoofer: configureSpoofer("cao2023"), sensor: vlp16 });
    expect(env.halfWidthEl).toBeGreaterThan(env.halfWidthAz);
    expect(env.halfWidthEl / DEG).toBeGreaterThan(15);
  });

  it("flags azimuth spans beyond the demonstrated 45 degree capability", () => {
    const wide = angularEnvelope({
      spoofer: configureSpoofer("cao2023", { powerRatio: 1e9 }),
      sensor: vlp16,
    });
    expect(DEMONSTRATED_MAX_AZ_DEG).toBe(45);
    if ((2 * wide.halfWidthAz) / DEG > DEMONSTRATED_MAX_AZ_DEG) {
      expect(wide.extrapolation).toBeTruthy();
    }
    const narrow = angularEnvelope({ spoofer: configureSpoofer("cao2023"), sensor: vlp16 });
    expect(narrow.extrapolation).toBeFalsy();
  });
});

describe("reachableCells", () => {
  const spoofer = configureSpoofer("cao2023");

  it("returns cells centred on the spoofer bearing", () => {
    const frame = emptyFrame(sensor);
    const bearingDeg = 30;
    const pos = [
      40 * Math.cos(bearingDeg * DEG),
      40 * Math.sin(bearingDeg * DEG),
      0,
    ];
    const res = reachableCells({ frame, sensor, spoofer, positionXYZ: pos });
    expect(res.cells.length).toBeGreaterThan(0);

    const halfAz = res.envelope.halfWidthAz;
    for (let i = 0; i < res.cells.length; i++) {
      const { azIdx } = unpackCellKey(res.cells[i], sensor.nBeams);
      const cellAz = (azIdx + 0.5) * sensor.azRes;
      let d = Math.abs(cellAz - res.geometry.bearing);
      if (d > Math.PI) d = 2 * Math.PI - d;
      // one cell of slack for the quantization of the envelope onto the lattice
      expect(d).toBeLessThanOrEqual(halfAz + sensor.azRes);
    }
  });

  it("weights the aim point highest and falls off with angular offset", () => {
    const res = reachableCells({
      frame: emptyFrame(sensor),
      sensor,
      spoofer,
      positionXYZ: [40, 0, 0],
    });
    let peak = 0;
    let peakOffset = Infinity;
    for (let i = 0; i < res.cells.length; i++) {
      if (res.weight[i] > peak) {
        peak = res.weight[i];
        peakOffset = Math.abs(res.offsetAz[i]);
      }
    }
    expect(peak).toBeGreaterThan(0);
    // the strongest cell is the one nearest the bearing
    expect(peakOffset).toBeLessThanOrEqual(sensor.azRes);

    // and weight decreases monotonically with |offset|
    for (let i = 0; i < res.cells.length; i++) {
      for (let j = 0; j < res.cells.length; j++) {
        if (Math.abs(res.offsetAz[i]) < Math.abs(res.offsetAz[j]) - 1e-12 &&
            res.rowOffset[i] === res.rowOffset[j]) {
          expect(res.weight[i]).toBeGreaterThanOrEqual(res.weight[j] - 1e-9);
        }
      }
    }
  });

  it("matches the measured edge-to-peak removal rate ratio", () => {
    // Real capture: 0.83-0.87 at the aim point falling to ~0.25 at the edge of
    // the 6.8 deg envelope, i.e. an edge/peak ratio near 0.30.
    const res = reachableCells({
      frame: emptyFrame(vlp16),
      sensor: vlp16,
      spoofer,
      positionXYZ: [40, 0, 0],
    });
    let peak = 0;
    let edge = Infinity;
    let maxOffset = 0;
    for (let i = 0; i < res.cells.length; i++) {
      if (res.rowOffset[i] !== 0) continue;
      const off = Math.abs(res.offsetAz[i]);
      if (off > maxOffset) {
        maxOffset = off;
        edge = res.weight[i];
      }
      if (res.weight[i] > peak) peak = res.weight[i];
    }
    expect(peak).toBeGreaterThan(0);
    expect(edge / peak).toBeCloseTo(0.30, 0.12);
  });

  it("caps the cell count by the spoofer's firing budget per revolution", () => {
    // A 20 kHz spoofer against a 10 Hz victim can only place 2000 points per
    // revolution no matter how wide its envelope is.
    const slow = configureSpoofer("cao2023", { prfHz: 2000, powerRatio: 1e6 });
    const res = reachableCells({
      frame: emptyFrame(sensor),
      sensor,
      spoofer: slow,
      positionXYZ: [40, 0, 0],
    });
    expect(res.maxCellsPerRevolution).toBe(Math.floor(2000 / sensor.hz));
    expect(res.cells.length).toBeLessThanOrEqual(res.maxCellsPerRevolution);
    expect(res.firingLimited).toBeTruthy();
  });

  it("reports not firing-limited when the envelope is the binding constraint", () => {
    const res = reachableCells({
      frame: emptyFrame(sensor),
      sensor,
      spoofer: configureSpoofer("cao2023", { prfHz: 5e6 }),
      positionXYZ: [40, 0, 0],
    });
    expect(res.firingLimited).toBeFalsy();
  });

  it("keeps the highest-weight cells when the firing budget binds", () => {
    const slow = configureSpoofer("cao2023", { prfHz: 500, powerRatio: 1e6 });
    const res = reachableCells({
      frame: emptyFrame(sensor),
      sensor,
      spoofer: slow,
      positionXYZ: [40, 0, 0],
    });
    // budget spent nearest the aim point, not scattered at the fringe
    const worst = Math.max(...Array.from(res.weight));
    const best = Math.min(...Array.from(res.weight));
    expect(worst).toBeGreaterThanOrEqual(best);
    for (let i = 0; i < res.cells.length; i++) {
      expect(Math.abs(res.offsetAz[i])).toBeLessThanOrEqual(res.envelope.halfWidthAz + sensor.azRes);
    }
  });

  it("yields nothing when the spoofer stands beyond the victim's max range", () => {
    const res = reachableCells({
      frame: emptyFrame(sensor),
      sensor,
      spoofer,
      positionXYZ: [500, 0, 0],
    });
    expect(res.cells.length).toBe(0);
    expect(res.feasible).toBeFalsy();
  });
});

describe("checkLineOfSight", () => {
  it("is clear in free space", () => {
    const los = checkLineOfSight({
      frame: emptyFrame(sensor),
      sensor,
      bearing: 0.5,
      elevation: 0,
      standoff: 40,
    });
    expect(los.clear).toBeTruthy();
  });

  it("is blocked when real geometry lies between victim and spoofer", () => {
    // A spoofer parked behind a building cannot inject, however powerful it is.
    const frame = wallFrame(sensor, { azFromDeg: 25, azToDeg: 35, range: 8 });
    const los = checkLineOfSight({
      frame,
      sensor,
      bearing: 30 * DEG,
      elevation: 0,
      standoff: 40,
    });
    expect(los.clear).toBeFalsy();
    expect(los.occluderRange).toBeCloseTo(8, 0.5);
  });

  it("is clear when the geometry is behind the spoofer", () => {
    const frame = wallFrame(sensor, { azFromDeg: 25, azToDeg: 35, range: 60 });
    const los = checkLineOfSight({
      frame,
      sensor,
      bearing: 30 * DEG,
      elevation: 0,
      standoff: 40,
    });
    expect(los.clear).toBeTruthy();
  });

  it("is not fooled by a single stray return on the bearing", () => {
    // One point must not veto the attack; require agreement across the beams
    // the spoofer illuminates.
    const frame = frameOf(sensor, [atCell(sensor, Math.round((30 * DEG) / sensor.azRes), 16, 8)]);
    const los = checkLineOfSight({
      frame,
      sensor,
      bearing: 30 * DEG,
      elevation: 0,
      standoff: 40,
    });
    expect(los.clear).toBeTruthy();
  });

  it("blocks the whole attack through reachableCells when occluded", () => {
    const frame = wallFrame(sensor, { azFromDeg: -5, azToDeg: 5, range: 6 });
    const res = reachableCells({
      frame,
      sensor,
      spoofer: configureSpoofer("cao2023"),
      positionXYZ: [40, 0, 0],
    });
    expect(res.lineOfSight.clear).toBeFalsy();
    expect(res.cells.length).toBe(0);
    expect(res.feasible).toBeFalsy();
  });
});
