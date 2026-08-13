/**
 * Minimal zero-dependency test harness.
 *
 * There is no JS runtime on this machine (no node/npm), so tests run in
 * headless Chrome and report by writing a machine-readable marker into the DOM
 * that run-tests.sh greps. Keep this dependency-free; importing a test file
 * only registers suites, `run()` executes them.
 */

const suites = [];
let current = null;

export function describe(name, fn) {
  const prev = current;
  current = { name, tests: [] };
  suites.push(current);
  try {
    fn();
  } finally {
    current = prev;
  }
}

export function it(name, fn) {
  if (!current) throw new Error('it("' + name + '") called outside describe()');
  current.tests.push({ name, fn });
}

/* ---------------- assertions ---------------- */

class AssertionError extends Error {
  constructor(msg) {
    super(msg);
    this.name = "AssertionError";
  }
}

function show(v) {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toPrecision(8);
  if (v instanceof Float32Array || v instanceof Float64Array || v instanceof Int32Array ||
      v instanceof Uint32Array || v instanceof Uint8Array || v instanceof Uint16Array) {
    const a = Array.from(v.slice(0, 12), (x) => show(x));
    return v.constructor.name + "[" + a.join(", ") + (v.length > 12 ? ", …" : "") + "]";
  }
  if (Array.isArray(v)) {
    const a = v.slice(0, 12).map(show);
    return "[" + a.join(", ") + (v.length > 12 ? ", …" : "") + "]";
  }
  if (v === null) return "null";
  if (v === undefined) return "undefined";
  if (typeof v === "object") {
    try {
      return JSON.stringify(v);
    } catch {
      return String(v);
    }
  }
  return JSON.stringify(v);
}

function isTyped(v) {
  return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a === "number" && typeof b === "number") {
    return Number.isNaN(a) && Number.isNaN(b);
  }
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") return false;
  const aArr = Array.isArray(a) || isTyped(a);
  const bArr = Array.isArray(b) || isTyped(b);
  if (aArr !== bArr) return false;
  if (aArr) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false;
    return true;
  }
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) {
    if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
    if (!deepEqual(a[k], b[k])) return false;
  }
  return true;
}

export function expect(actual) {
  const api = {
    toBe(exp) {
      if (actual !== exp) {
        throw new AssertionError("expected " + show(exp) + " but got " + show(actual));
      }
    },
    toEqual(exp) {
      if (!deepEqual(actual, exp)) {
        throw new AssertionError("expected " + show(exp) + " but got " + show(actual));
      }
    },
    /** Absolute tolerance — the right default for physical quantities. */
    toBeCloseTo(exp, eps = 1e-9) {
      if (typeof actual !== "number" || !Number.isFinite(actual)) {
        throw new AssertionError("expected a finite number close to " + show(exp) + ", got " + show(actual));
      }
      if (Math.abs(actual - exp) > eps) {
        throw new AssertionError(
          "expected " + show(exp) + " ± " + eps + " but got " + show(actual) +
          " (off by " + Math.abs(actual - exp).toPrecision(3) + ")"
        );
      }
    },
    toBeGreaterThan(exp) {
      if (!(actual > exp)) throw new AssertionError("expected " + show(actual) + " > " + show(exp));
    },
    toBeGreaterThanOrEqual(exp) {
      if (!(actual >= exp)) throw new AssertionError("expected " + show(actual) + " >= " + show(exp));
    },
    toBeLessThan(exp) {
      if (!(actual < exp)) throw new AssertionError("expected " + show(actual) + " < " + show(exp));
    },
    toBeLessThanOrEqual(exp) {
      if (!(actual <= exp)) throw new AssertionError("expected " + show(actual) + " <= " + show(exp));
    },
    toBeTruthy() {
      if (!actual) throw new AssertionError("expected truthy, got " + show(actual));
    },
    toBeFalsy() {
      if (actual) throw new AssertionError("expected falsy, got " + show(actual));
    },
    toHaveLength(n) {
      if (actual == null || actual.length !== n) {
        throw new AssertionError("expected length " + n + " but got " + (actual == null ? "null" : actual.length));
      }
    },
    toContain(needle) {
      const ok = isTyped(actual) ? Array.from(actual).includes(needle) : actual.includes(needle);
      if (!ok) throw new AssertionError("expected " + show(actual) + " to contain " + show(needle));
    },
    /** Elementwise closeness for arrays / typed arrays. */
    toBeCloseToArray(exp, eps = 1e-9) {
      if (actual == null || actual.length !== exp.length) {
        throw new AssertionError(
          "expected length " + exp.length + " but got " + (actual == null ? "null" : actual.length)
        );
      }
      for (let i = 0; i < exp.length; i++) {
        if (!(Math.abs(actual[i] - exp[i]) <= eps)) {
          throw new AssertionError(
            "at index " + i + ": expected " + show(exp[i]) + " ± " + eps + " but got " + show(actual[i]) +
            "\n  full actual: " + show(actual) + "\n  full expect: " + show(exp)
          );
        }
      }
    },
    toThrow(match) {
      let threw = null;
      try {
        actual();
      } catch (e) {
        threw = e;
      }
      if (!threw) throw new AssertionError("expected function to throw, but it did not");
      if (match && !String(threw.message).includes(match)) {
        throw new AssertionError('expected thrown message to include "' + match + '" but got "' + threw.message + '"');
      }
    },
  };
  api.not = {
    toBe(exp) {
      if (actual === exp) throw new AssertionError("expected value NOT to be " + show(exp));
    },
    toEqual(exp) {
      if (deepEqual(actual, exp)) throw new AssertionError("expected value NOT to equal " + show(exp));
    },
    toThrow() {
      try {
        actual();
      } catch (e) {
        throw new AssertionError("expected function not to throw, but it threw: " + e.message);
      }
    },
  };
  return api;
}

/* ---------------- runner ---------------- */

export async function run() {
  const results = [];
  let pass = 0;
  let fail = 0;

  for (const suite of suites) {
    for (const t of suite.tests) {
      const label = suite.name + " › " + t.name;
      try {
        await t.fn();
        pass++;
        results.push({ ok: true, label });
      } catch (e) {
        fail++;
        results.push({ ok: false, label, error: e, stack: e && e.stack ? e.stack : "" });
      }
    }
  }
  return { results, pass, fail, total: pass + fail };
}

/** Run every registered suite and write a report the CLI can parse. */
export async function runAndReport() {
  let summary;
  try {
    summary = await run();
  } catch (e) {
    summary = { results: [{ ok: false, label: "harness", error: e }], pass: 0, fail: 1, total: 1 };
  }

  const lines = [];
  for (const r of summary.results) {
    if (r.ok) {
      lines.push("  pass  " + r.label);
    } else {
      lines.push("  FAIL  " + r.label);
      const msg = r.error && r.error.message ? r.error.message : String(r.error);
      for (const l of String(msg).split("\n")) lines.push("          " + l);
      if (r.error && r.error.name !== "AssertionError" && r.stack) {
        const frame = String(r.stack).split("\n").slice(1, 3).join("\n");
        for (const l of frame.split("\n")) lines.push("          " + l.trim());
      }
    }
  }
  lines.push("");
  lines.push(summary.fail === 0
    ? "ALL PASSING — " + summary.pass + " tests"
    : summary.fail + " FAILING, " + summary.pass + " passing");

  const pre = document.createElement("pre");
  pre.id = "atlas-results";
  pre.textContent = lines.join("\n");
  document.body.appendChild(pre);

  // Machine-readable marker: run-tests.sh greps for this exact shape.
  const marker = document.createElement("div");
  marker.id = "atlas-done";
  marker.textContent = "ATLAS_TESTS_DONE pass=" + summary.pass + " fail=" + summary.fail;
  document.body.appendChild(marker);

  document.title = summary.fail === 0 ? "PASS" : "FAIL";
  return summary;
}
