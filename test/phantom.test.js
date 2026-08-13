import { describe, it, expect } from "./harness.js";
import { configureSensor } from "../src/core/sensor.js";
import { configureSpoofer, reachableCells } from "../src/core/spoofer.js";
import { emptyFrame } from "./fixtures.js";
import { rayBox, boxPhantom, wallPhantom, PHANTOM_SHAPES } from "../src/core/phantom.js";
import { cartToSpherical } from "../src/core/polar.js";

const sensor = configureSensor("hdl32e");
const spoofer = configureSpoofer("cao2023");
const reach = reachableCells({
  frame: emptyFrame(sensor),
  sensor,
  spoofer,
  positionXYZ: [40, 0, 0],
});

describe("rayBox", () => {
  const box = { center: [60, 0, 0], size: [4, 2, 1.6], yaw: 0 };

  it("hits the near face of a box straight ahead", () => {
    const t = rayBox([1, 0, 0], box);
    expect(t).toBeCloseTo(58, 1e-6); // 60 - 4/2
  });

  it("misses when the ray points away", () => {
    expect(rayBox([-1, 0, 0], box)).toBe(Infinity);
    expect(rayBox([0, 1, 0], box)).toBe(Infinity);
  });

  it("misses when the ray passes beside the box", () => {
    const dir = cartToSpherical(60, 20, 0);
    const d = [Math.cos(dir.az), Math.sin(dir.az), 0];
    expect(rayBox(d, box)).toBe(Infinity);
  });

  it("respects yaw when orienting the box", () => {
    const rotated = { center: [60, 0, 0], size: [8, 1, 1.6], yaw: Math.PI / 2 };
    // rotated 90 deg, the long axis is now across the ray, so the near face is
    // at 60 - 0.5 rather than 60 - 4
    expect(rayBox([1, 0, 0], rotated)).toBeCloseTo(59.5, 1e-6);
  });

  it("returns the nearer of the two slab crossings", () => {
    const t = rayBox([1, 0, 0], box);
    expect(t).toBeLessThan(60);
  });
});

describe("boxPhantom", () => {
  const box = { center: [60, 0, -0.7], size: [4.2, 1.8, 1.6], yaw: 0 };

  it("only produces points on cells the spoofer can actually reach", () => {
    const p = boxPhantom({ box, sensor, reach });
    expect(p.x.length).toBeGreaterThan(0);
    const reachable = new Set(Array.from(reach.cells));
    for (let i = 0; i < p.x.length; i++) {
      expect(reachable.has(p.cell[i])).toBeTruthy();
    }
  });

  it("gives at most one point per cell, so it is realizable by construction", () => {
    const p = boxPhantom({ box, sensor, reach });
    expect(new Set(Array.from(p.cell)).size).toBe(p.cell.length);
  });

  it("places every point on the surface facing the sensor", () => {
    const p = boxPhantom({ box, sensor, reach });
    for (let i = 0; i < p.x.length; i++) {
      const r = Math.hypot(p.x[i], p.y[i], p.z[i]);
      // near face of a 4.2 m box centred at 60 m
      expect(r).toBeGreaterThan(57);
      expect(r).toBeLessThan(60.5);
    }
  });

  it("produces nothing for a box outside the reachable wedge", () => {
    const aside = { center: [0, 60, -0.7], size: [4.2, 1.8, 1.6], yaw: 0 };
    const p = boxPhantom({ box: aside, sensor, reach });
    expect(p.x.length).toBe(0);
  });

  it("scales point count with apparent size", () => {
    const small = boxPhantom({ box: { ...box, size: [1, 1, 0.6] }, sensor, reach });
    const large = boxPhantom({ box: { ...box, size: [6, 4, 2.4] }, sensor, reach });
    expect(large.x.length).toBeGreaterThan(small.x.length);
  });

  it("puts a nearer box at a shorter range", () => {
    const near = boxPhantom({ box: { ...box, center: [50, 0, -0.7] }, sensor, reach });
    const far = boxPhantom({ box: { ...box, center: [80, 0, -0.7] }, sensor, reach });
    const meanR = (p) => {
      let s = 0;
      for (let i = 0; i < p.x.length; i++) s += Math.hypot(p.x[i], p.y[i], p.z[i]);
      return s / p.x.length;
    };
    expect(meanR(near)).toBeLessThan(meanR(far));
  });
});

describe("wallPhantom", () => {
  it("fills the whole reachable envelope at one range", () => {
    const p = wallPhantom({ range: 55, sensor, reach });
    expect(p.x.length).toBe(reach.cells.length);
    for (let i = 0; i < p.x.length; i++) {
      expect(Math.hypot(p.x[i], p.y[i], p.z[i])).toBeCloseTo(55, 1e-4);
    }
  });

  it("is still one point per cell", () => {
    const p = wallPhantom({ range: 55, sensor, reach });
    expect(new Set(Array.from(p.cell)).size).toBe(p.cell.length);
  });
});

describe("PHANTOM_SHAPES", () => {
  it("lists the shapes the UI offers", () => {
    expect(PHANTOM_SHAPES).toContain("box");
    expect(PHANTOM_SHAPES).toContain("wall");
  });
});
