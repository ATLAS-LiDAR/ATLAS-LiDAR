import { describe, it, expect } from "./harness.js";
import {
  normalizeBox,
  bevCorners,
  polygonArea,
  convexIntersection,
  bevIoU,
  iou3d,
  maxIoU,
} from "../src/core/iou.js";

const box = (x, y, z, dx, dy, dz, yaw = 0) => ({ center: [x, y, z], size: [dx, dy, dz], yaw });

describe("normalizeBox", () => {
  it("accepts the pcdet 7-vector [x,y,z,dx,dy,dz,heading]", () => {
    const b = normalizeBox([1, 2, 3, 4, 2, 1.6, 0.5]);
    expect(b.center).toEqual([1, 2, 3]);
    expect(b.size).toEqual([4, 2, 1.6]);
    expect(b.yaw).toBeCloseTo(0.5, 1e-12);
  });

  it("accepts the object form used by the frame exporter", () => {
    const b = normalizeBox(box(1, 2, 3, 4, 2, 1.6, 0.5));
    expect(b.center).toEqual([1, 2, 3]);
    expect(b.yaw).toBeCloseTo(0.5, 1e-12);
  });
});

describe("bevCorners and polygonArea", () => {
  it("returns four corners enclosing the stated footprint", () => {
    const c = bevCorners(normalizeBox(box(0, 0, 0, 4, 2, 1)));
    expect(c).toHaveLength(4);
    expect(Math.abs(polygonArea(c))).toBeCloseTo(8, 1e-9);
  });

  it("preserves area under rotation", () => {
    const c = bevCorners(normalizeBox(box(3, -1, 0, 4, 2, 1, 0.7)));
    expect(Math.abs(polygonArea(c))).toBeCloseTo(8, 1e-9);
  });
});

describe("convexIntersection", () => {
  it("clips one square against another", () => {
    const a = bevCorners(normalizeBox(box(0, 0, 0, 2, 2, 1)));
    const b = bevCorners(normalizeBox(box(1, 0, 0, 2, 2, 1)));
    const p = convexIntersection(a, b);
    expect(Math.abs(polygonArea(p))).toBeCloseTo(2, 1e-9); // 1 x 2 overlap
  });

  it("returns the inner polygon when one contains the other", () => {
    const outer = bevCorners(normalizeBox(box(0, 0, 0, 10, 10, 1)));
    const inner = bevCorners(normalizeBox(box(0, 0, 0, 2, 3, 1)));
    expect(Math.abs(polygonArea(convexIntersection(outer, inner)))).toBeCloseTo(6, 1e-9);
    expect(Math.abs(polygonArea(convexIntersection(inner, outer)))).toBeCloseTo(6, 1e-9);
  });

  it("returns empty for disjoint polygons", () => {
    const a = bevCorners(normalizeBox(box(0, 0, 0, 2, 2, 1)));
    const b = bevCorners(normalizeBox(box(9, 9, 0, 2, 2, 1)));
    expect(convexIntersection(a, b).length).toBe(0);
  });
});

describe("bevIoU", () => {
  it("is 1 for identical boxes", () => {
    const a = box(1, 2, 0, 4, 2, 1.6, 0.3);
    expect(bevIoU(normalizeBox(a), normalizeBox(a))).toBeCloseTo(1, 1e-9);
  });

  it("is 0 for disjoint boxes", () => {
    expect(bevIoU(normalizeBox(box(0, 0, 0, 2, 2, 1)), normalizeBox(box(50, 0, 0, 2, 2, 1)))).toBe(0);
  });

  it("matches the analytic value for a half-overlap", () => {
    // two 2x2 squares offset by 1 in x: intersection 2, union 6 -> 1/3
    const v = bevIoU(normalizeBox(box(0, 0, 0, 2, 2, 1)), normalizeBox(box(1, 0, 0, 2, 2, 1)));
    expect(v).toBeCloseTo(1 / 3, 1e-9);
  });

  it("is symmetric", () => {
    const a = normalizeBox(box(0, 0, 0, 4, 2, 1, 0.4));
    const b = normalizeBox(box(1, 0.5, 0, 3, 2.5, 1, -0.2));
    expect(bevIoU(a, b)).toBeCloseTo(bevIoU(b, a), 1e-12);
  });

  it("is invariant to a 90 degree yaw on a square footprint", () => {
    const a = normalizeBox(box(0, 0, 0, 2, 2, 1, 0));
    const b = normalizeBox(box(0, 0, 0, 2, 2, 1, Math.PI / 2));
    expect(bevIoU(a, b)).toBeCloseTo(1, 1e-9);
  });

  it("drops when a box is rotated off alignment", () => {
    const a = normalizeBox(box(0, 0, 0, 4, 2, 1, 0));
    const b = normalizeBox(box(0, 0, 0, 4, 2, 1, Math.PI / 2));
    const v = bevIoU(a, b);
    // the intersection of the two is a 2x2 square: 4 / (8 + 8 - 4) = 1/3
    expect(v).toBeCloseTo(1 / 3, 1e-6);
  });
});

describe("iou3d", () => {
  it("equals BEV IoU when the boxes share the same vertical extent", () => {
    const a = normalizeBox(box(0, 0, 0, 2, 2, 2));
    const b = normalizeBox(box(1, 0, 0, 2, 2, 2));
    expect(iou3d(a, b)).toBeCloseTo(bevIoU(a, b), 1e-9);
  });

  it("is 0 when the boxes do not overlap in z", () => {
    const a = normalizeBox(box(0, 0, 0, 2, 2, 1));
    const b = normalizeBox(box(0, 0, 5, 2, 2, 1));
    expect(iou3d(a, b)).toBe(0);
  });

  it("matches the analytic value for unit cubes offset in x", () => {
    // intersection 0.5, union 2 - 0.5 = 1.5 -> 1/3
    const a = normalizeBox(box(0, 0, 0, 1, 1, 1));
    const b = normalizeBox(box(0.5, 0, 0, 1, 1, 1));
    expect(iou3d(a, b)).toBeCloseTo(1 / 3, 1e-9);
  });

  it("accounts for partial vertical overlap", () => {
    // same footprint, halves overlapping in z: inter 0.5, union 1.5 -> 1/3
    const a = normalizeBox(box(0, 0, 0, 1, 1, 1));
    const b = normalizeBox(box(0, 0, 0.5, 1, 1, 1));
    expect(iou3d(a, b)).toBeCloseTo(1 / 3, 1e-9);
  });

  it("is 1 for identical boxes and symmetric in general", () => {
    const a = normalizeBox(box(2, 3, -1, 4.2, 1.8, 1.6, 0.9));
    expect(iou3d(a, a)).toBeCloseTo(1, 1e-9);
    const b = normalizeBox(box(2.5, 3.2, -0.8, 3.9, 2.0, 1.5, 0.7));
    expect(iou3d(a, b)).toBeCloseTo(iou3d(b, a), 1e-12);
  });

  it("never exceeds 1 or falls below 0", () => {
    const a = normalizeBox(box(0, 0, 0, 4, 2, 1.6, 0.3));
    for (let i = 0; i < 40; i++) {
      const b = normalizeBox(box(i * 0.2 - 4, i * 0.1 - 2, i * 0.05 - 1, 4, 2, 1.6, i * 0.15));
      const v = iou3d(a, b);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("maxIoU", () => {
  it("returns the best match and its index", () => {
    const target = normalizeBox(box(10, 0, 0, 4, 2, 1.6));
    const preds = [box(50, 0, 0, 4, 2, 1.6), box(10.2, 0.1, 0, 4, 2, 1.6), box(30, 0, 0, 4, 2, 1.6)];
    const r = maxIoU(preds, target);
    expect(r.index).toBe(1);
    expect(r.iou).toBeGreaterThan(0.8);
  });

  it("returns iou 0 and index -1 for an empty prediction set", () => {
    // This case decides an ATLAS success: no predictions at all means a removal
    // attack succeeded and an injection attack failed.
    const r = maxIoU([], normalizeBox(box(10, 0, 0, 4, 2, 1.6)));
    expect(r.iou).toBe(0);
    expect(r.index).toBe(-1);
  });
});
