/**
 * Tests for the harness itself. A test runner that cannot report failure is
 * worse than no runner, so these assert that the matchers actually throw.
 */
import { describe, it, expect } from "./harness.js";

function throws(fn) {
  try {
    fn();
  } catch (e) {
    return e;
  }
  return null;
}

describe("harness matchers", () => {
  it("toBe passes on identity and fails otherwise", () => {
    expect(1).toBe(1);
    expect(throws(() => expect(1).toBe(2))).toBeTruthy();
  });

  it("toBeCloseTo respects the tolerance in both directions", () => {
    expect(1.0000000001).toBeCloseTo(1.0, 1e-6);
    expect(throws(() => expect(1.1).toBeCloseTo(1.0, 1e-6))).toBeTruthy();
    // NaN must never silently pass a closeness check
    expect(throws(() => expect(NaN).toBeCloseTo(1.0, 1e6))).toBeTruthy();
  });

  it("toEqual compares arrays and typed arrays structurally", () => {
    expect([1, 2, 3]).toEqual([1, 2, 3]);
    expect(new Float32Array([1, 2])).toEqual(new Float32Array([1, 2]));
    expect(throws(() => expect([1, 2]).toEqual([1, 2, 3]))).toBeTruthy();
    expect({ a: 1, b: [2] }).toEqual({ a: 1, b: [2] });
    expect(throws(() => expect({ a: 1 }).toEqual({ a: 1, b: 2 }))).toBeTruthy();
  });

  it("toBeCloseToArray reports the offending index", () => {
    expect(new Float64Array([1, 2])).toBeCloseToArray([1, 2], 1e-9);
    const e = throws(() => expect([1, 5]).toBeCloseToArray([1, 2], 1e-9));
    expect(e).toBeTruthy();
    expect(e.message).toContain("at index 1");
  });

  it("toThrow detects throwing and non-throwing functions", () => {
    expect(() => {
      throw new Error("boom");
    }).toThrow("boom");
    expect(throws(() => expect(() => 1).toThrow())).toBeTruthy();
  });

  it("supports async test bodies", async () => {
    const v = await Promise.resolve(7);
    expect(v).toBe(7);
  });
});
