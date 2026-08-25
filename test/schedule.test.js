import { describe, it, expect } from "./harness.js";
import { buildAttackSchedule, SCHEDULE_DEFAULTS } from "../src/core/schedule.js";

const sched = (o) => buildAttackSchedule({ nFrames: 20, seed: 7, ...o });

describe("buildAttackSchedule — port of ATLAS build_single_attack_schedule", () => {
  it("exposes ATLAS's documented defaults", () => {
    expect(SCHEDULE_DEFAULTS.minLen).toBe(2);
    expect(SCHEDULE_DEFAULTS.maxLen).toBe(32);
    expect(SCHEDULE_DEFAULTS.duty).toBeCloseTo(0.5, 1e-12);
    expect(SCHEDULE_DEFAULTS.cleanPrefix).toBe(4);
  });

  it("gives every sequence exactly one attack", () => {
    for (let seed = 1; seed <= 20; seed++) {
      const s = sched({ seed, mode: "continuous" });
      expect(s.meta.attackLenL).toBeGreaterThan(0);
      expect(s.attacked.length).toBe(s.meta.attackLenL);
    }
  });

  it("keeps the opening frames clean", () => {
    for (let seed = 1; seed <= 20; seed++) {
      for (const mode of ["continuous", "intermittent"]) {
        const s = sched({ seed, mode });
        for (const f of s.attacked) expect(f).toBeGreaterThanOrEqual(SCHEDULE_DEFAULTS.cleanPrefix);
      }
    }
  });

  it("never schedules a frame outside the sequence", () => {
    for (let seed = 1; seed <= 30; seed++) {
      for (const mode of ["continuous", "intermittent"]) {
        const s = sched({ seed, mode, nFrames: 12 });
        for (const f of s.attacked) {
          expect(f).toBeGreaterThanOrEqual(0);
          expect(f).toBeLessThan(12);
        }
      }
    }
  });

  it("continuous mode attacks a contiguous run", () => {
    for (let seed = 1; seed <= 20; seed++) {
      const s = sched({ seed, mode: "continuous" });
      const a = [...s.attacked].sort((x, y) => x - y);
      for (let i = 1; i < a.length; i++) expect(a[i]).toBe(a[i - 1] + 1);
      expect(s.meta.nGaps).toBe(0);
      expect(s.meta.spanLen).toBe(s.meta.attackLenL);
    }
  });

  it("intermittent mode spreads the SAME number of attacked frames over a longer span", () => {
    // ATLAS holds the attacked-frame count equal to the continuous case at the
    // same L, so the two modes differ only in continuity.
    for (let seed = 1; seed <= 20; seed++) {
      const c = sched({ seed, mode: "continuous" });
      const i = sched({ seed, mode: "intermittent" });
      expect(i.meta.requestedL).toBe(c.meta.requestedL);
      if (!i.meta.degenerate) {
        expect(i.meta.spanLen).toBeGreaterThan(i.meta.attackLenL);
        expect(i.meta.nGaps).toBeGreaterThan(0);
      }
    }
  });

  it("keeps the span endpoints attacked so the span is well defined", () => {
    for (let seed = 1; seed <= 20; seed++) {
      const s = sched({ seed, mode: "intermittent" });
      if (s.meta.degenerate || s.attacked.length < 2) continue;
      const a = [...s.attacked].sort((x, y) => x - y);
      expect(a[0]).toBe(s.meta.spanStart);
      expect(a[a.length - 1]).toBe(s.meta.spanStart + s.meta.spanLen - 1);
    }
  });

  it("flags a degenerate intermittent attack instead of miscounting it", () => {
    // When L fills the whole available span there is no room for a gap, so ATLAS
    // marks it degenerate rather than silently calling it intermittent.
    const s = buildAttackSchedule({ nFrames: 6, seed: 3, mode: "intermittent", minLen: 2, maxLen: 32 });
    if (s.meta.spanLen <= s.meta.attackLenL) {
      expect(s.meta.degenerate).toBeTruthy();
      expect(s.meta.nGaps).toBe(0);
    }
  });

  // WHY THE SEQUENCE ASSETS ARE FULL-LENGTH
  // ATLAS's maxLen of 32 was chosen against full nuScenes scenes, which are 20 s
  // at 2 Hz = 40 keyframes. With cleanPrefix 4 that leaves avail = 36 > 32, so L
  // can never reach the available span and intermittent mode always has room for
  // a gap. Truncating the export to 20 keyframes leaves avail = 16, and any
  // requestedL >= 16 collapses to degenerate - over half of all seeds. The fix is
  // the asset length, not the algorithm, so this pins both directions.
  it("never degenerates at full nuScenes scene length", () => {
    for (let seed = 1; seed <= 200; seed++) {
      const s = buildAttackSchedule({ nFrames: 40, seed, mode: "intermittent", label: "s" });
      expect(s.meta.degenerate).toBeFalsy();
      expect(s.meta.nGaps).toBeGreaterThan(0);
      expect(s.meta.spanLen).toBeGreaterThan(s.meta.attackLenL);
    }
  });

  it("degenerates for most seeds at HALF a scene, which is why the export is not truncated", () => {
    let bad = 0;
    for (let seed = 1; seed <= 200; seed++) {
      if (buildAttackSchedule({ nFrames: 20, seed, mode: "intermittent", label: "s" }).meta.degenerate) {
        bad++;
      }
    }
    // requestedL ~ uniform{2..32}; P(requestedL >= 16) = 17/31 ~ 0.55
    expect(bad).toBeGreaterThan(80);
  });

  it("clamps the attack length to the frames actually available", () => {
    const s = sched({ nFrames: 8, mode: "continuous", minLen: 30, maxLen: 32 });
    // 8 frames minus a 4-frame clean prefix leaves 4
    expect(s.meta.attackLenL).toBeLessThanOrEqual(4);
  });

  it("reports the realised duty, since it cannot always be the requested one", () => {
    const s = sched({ mode: "intermittent" });
    expect(s.meta.realisedDuty).toBeGreaterThan(0);
    expect(s.meta.realisedDuty).toBeLessThanOrEqual(1);
    expect(s.meta.realisedDuty).toBeCloseTo(s.meta.attackLenL / s.meta.spanLen, 1e-9);
  });

  it("lists the gap frames inside the span", () => {
    const s = sched({ mode: "intermittent" });
    for (const g of s.meta.gapLocal) {
      expect(g).toBeGreaterThanOrEqual(s.meta.spanStart);
      expect(g).toBeLessThan(s.meta.spanStart + s.meta.spanLen);
      expect(s.attacked.includes(g)).toBeFalsy();
    }
    expect(s.meta.gapLocal.length).toBe(s.meta.nGaps);
  });

  it("provides isAttacked for the frame the UI is showing", () => {
    const s = sched({ mode: "continuous" });
    for (let f = 0; f < 20; f++) {
      expect(s.isAttacked(f)).toBe(s.attacked.includes(f));
    }
  });

  it("is reproducible for a seed and varies across seeds", () => {
    const a = sched({ seed: 21, mode: "intermittent" });
    const b = sched({ seed: 21, mode: "intermittent" });
    expect(a.attacked).toEqual(b.attacked);
    let differs = false;
    for (let seed = 22; seed < 40; seed++) {
      const c = sched({ seed, mode: "intermittent" });
      if (JSON.stringify(c.attacked) !== JSON.stringify(a.attacked)) differs = true;
    }
    expect(differs).toBeTruthy();
  });

  it("rejects an unknown mode instead of silently defaulting", () => {
    expect(() => sched({ mode: "sometimes" })).toThrow("sometimes");
  });

  it("handles a sequence too short to attack", () => {
    const s = buildAttackSchedule({ nFrames: 3, seed: 1, mode: "continuous", cleanPrefix: 4 });
    expect(s.attacked.length).toBeGreaterThanOrEqual(0);
    for (const f of s.attacked) expect(f).toBeLessThan(3);
  });
});
