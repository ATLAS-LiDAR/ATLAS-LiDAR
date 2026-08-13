import { describe, it, expect } from "./harness.js";
import { rng, uniform } from "./util.js";
import {
  cartToSpherical,
  sphericalToCart,
  azimuthCellIndex,
  azimuthCellCount,
  nearestBeamIndex,
  cellKey,
  unpackCellKey,
  TWO_PI,
} from "../src/core/polar.js";

const DEG = Math.PI / 180;

describe("cartToSpherical", () => {
  it("uses the [0, 2pi) azimuth convention, not atan2's signed range", () => {
    // This matters: ATLAS indexes azimuth cells as floor(theta / dtheta), which
    // is only correct if theta is wrapped into [0, 2pi) first (the `% 2*pi` in
    // nuscenes_atlas_common.cart_2_spherical).
    expect(cartToSpherical(1, 0, 0).az).toBeCloseTo(0, 1e-12);
    expect(cartToSpherical(0, 1, 0).az).toBeCloseTo(Math.PI / 2, 1e-12);
    expect(cartToSpherical(-1, 0, 0).az).toBeCloseTo(Math.PI, 1e-12);
    expect(cartToSpherical(0, -1, 0).az).toBeCloseTo(3 * Math.PI / 2, 1e-12);
  });

  it("reports range as full 3D distance and elevation as asin(z/r)", () => {
    const s = cartToSpherical(3, 4, 0);
    expect(s.r).toBeCloseTo(5, 1e-12);
    expect(s.el).toBeCloseTo(0, 1e-12);

    const up = cartToSpherical(0, 0, 7);
    expect(up.r).toBeCloseTo(7, 1e-12);
    expect(up.el).toBeCloseTo(Math.PI / 2, 1e-12);

    const down = cartToSpherical(1, 0, -1);
    expect(down.el).toBeCloseTo(-Math.PI / 4, 1e-12);
  });

  it("does not produce NaN at the origin", () => {
    const s = cartToSpherical(0, 0, 0);
    expect(s.r).toBeCloseTo(0, 1e-12);
    expect(Number.isNaN(s.el)).toBeFalsy();
    expect(Number.isNaN(s.az)).toBeFalsy();
  });

  it("round-trips through sphericalToCart", () => {
    const next = rng(7);
    for (let i = 0; i < 500; i++) {
      const x = uniform(next, -80, 80);
      const y = uniform(next, -80, 80);
      const z = uniform(next, -6, 12);
      const { r, az, el } = cartToSpherical(x, y, z);
      const back = sphericalToCart(r, az, el);
      expect(back).toBeCloseToArray([x, y, z], 1e-9);
    }
  });
});

describe("azimuthCellIndex", () => {
  const dtheta = 0.33159202 * DEG; // real nuScenes azimuth resolution

  it("floors into cells and starts at zero", () => {
    expect(azimuthCellIndex(0, dtheta)).toBe(0);
    expect(azimuthCellIndex(dtheta * 0.999, dtheta)).toBe(0);
    expect(azimuthCellIndex(dtheta, dtheta)).toBe(1);
    expect(azimuthCellIndex(dtheta * 2.5, dtheta)).toBe(2);
  });

  it("wraps negative and >=2pi azimuths into range", () => {
    expect(azimuthCellIndex(-dtheta * 0.5, dtheta)).toBe(azimuthCellCount(dtheta) - 1);
    expect(azimuthCellIndex(TWO_PI, dtheta)).toBe(0);
    expect(azimuthCellIndex(TWO_PI + dtheta * 1.5, dtheta)).toBe(1);
  });

  it("never returns an index outside [0, cellCount)", () => {
    // A cell index equal to cellCount would alias onto cell 0 downstream and
    // silently corrupt the occupancy grid.
    const n = azimuthCellCount(dtheta);
    const next = rng(11);
    for (let i = 0; i < 2000; i++) {
      const az = uniform(next, -20, 20);
      const idx = azimuthCellIndex(az, dtheta);
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(n);
    }
    // the exact upper boundary is the dangerous case
    expect(azimuthCellIndex(TWO_PI - 1e-15, dtheta)).toBeLessThan(n);
  });
});

describe("nearestBeamIndex", () => {
  // ascending, irregularly spaced, like real hardware
  const inc = [-0.4, -0.3, -0.21, -0.1, -0.02, 0.05, 0.14, 0.3];

  it("returns the exact index for an exact inclination", () => {
    for (let i = 0; i < inc.length; i++) {
      expect(nearestBeamIndex(inc[i], inc)).toBe(i);
    }
  });

  it("clamps beyond either end of the vertical FOV", () => {
    expect(nearestBeamIndex(-2.0, inc)).toBe(0);
    expect(nearestBeamIndex(2.0, inc)).toBe(inc.length - 1);
  });

  it("picks the nearer neighbour across each gap", () => {
    expect(nearestBeamIndex(-0.31, inc)).toBe(1);
    expect(nearestBeamIndex(-0.25, inc)).toBe(2); // 0.04 from -0.21 vs 0.05 from -0.30
    expect(nearestBeamIndex(0.0, inc)).toBe(4);
  });

  it("agrees with brute-force argmin on random elevations", () => {
    // Guards the binary search against off-by-one, which is the whole risk of
    // not writing the O(n) version.
    const next = rng(23);
    for (let i = 0; i < 3000; i++) {
      const el = uniform(next, -0.6, 0.6);
      let best = 0;
      let bestD = Infinity;
      for (let j = 0; j < inc.length; j++) {
        const d = Math.abs(inc[j] - el);
        if (d < bestD) {
          bestD = d;
          best = j;
        }
      }
      expect(nearestBeamIndex(el, inc)).toBe(best);
    }
  });

  it("works on a Float64Array of real 64-beam inclinations", () => {
    const real = new Float64Array([-0.418661, -0.393843, -0.380018, -0.370361, -0.358113, 0.108056]);
    expect(nearestBeamIndex(-0.42, real)).toBe(0);
    expect(nearestBeamIndex(0.1, real)).toBe(5);
  });
});

describe("cellKey", () => {
  it("packs and unpacks losslessly", () => {
    const nBeams = 64;
    const k = cellKey(1523, 37, nBeams);
    expect(unpackCellKey(k, nBeams)).toEqual({ azIdx: 1523, beamIdx: 37 });
  });

  it("is collision-free across the full grid", () => {
    // ATLAS uses az_idx * 1000 + beam_idx, which is only safe while nBeams <
    // 1000; assert our packing has no such hidden ceiling for realistic grids.
    const nBeams = 128;
    const seen = new Set();
    for (let a = 0; a < 400; a++) {
      for (let b = 0; b < nBeams; b++) {
        const k = cellKey(a, b, nBeams);
        expect(seen.has(k)).toBeFalsy();
        seen.add(k);
      }
    }
    expect(seen.size).toBe(400 * nBeams);
  });
});
