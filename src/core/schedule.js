/**
 * Temporal attack schedule, ported from ATLAS's build_single_attack_schedule.
 *
 * Every sequence gets exactly ONE attack, of a length L that varies from
 * sequence to sequence. Two modes:
 *
 *   continuous   - L contiguous attacked frames.
 *   intermittent - the SAME attack toggling on and off: L attacked frames spread
 *                  over a longer span with clean frames between them. The
 *                  attacked-frame count is held equal to the continuous case at
 *                  the same L, so the two modes differ only in continuity.
 *
 * Two details worth preserving rather than smoothing:
 *
 *  - The duty cannot always be honoured. Intermittent needs span = L/duty, and a
 *    short sequence minus the clean prefix may not have room, so the duty degrades
 *    toward 1.0 as L approaches the available span. Sequences where L fills the
 *    span have no room for any gap and are flagged `degenerate` so analysis can
 *    exclude them instead of counting them as intermittent.
 *  - The span's endpoints are always attacked, so the span is well defined; the
 *    remaining attacked frames are spread evenly through the interior.
 *
 * ATLAS derives its per-scene rng from a sha256 of (seed, scene name) because
 * Python's hash() is salted per process. Here the sequence identity is folded in
 * with a plain string hash: the numpy stream cannot be reproduced in JS anyway,
 * so what matters is that the same (seed, sequence) always gives the same
 * schedule, which it does.
 */

export const SCHEDULE_DEFAULTS = Object.freeze({
  minLen: 2,
  maxLen: 32,
  duty: 0.5,
  cleanPrefix: 4,
});

const MODES = ["continuous", "intermittent"];

function hashSeed(seed, label) {
  let h = (seed >>> 0) ^ 0x9e3779b9;
  const s = String(label ?? "");
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0);
  }
  return h >>> 0;
}

function makeRng(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** numpy's rng.integers(lo, hi) semantics: inclusive lo, exclusive hi. */
const integers = (next, lo, hi) => lo + Math.floor(next() * Math.max(1, hi - lo));

/**
 * @returns {{attacked: number[], isAttacked: (i:number)=>boolean, meta: object}}
 */
export function buildAttackSchedule({
  nFrames,
  mode = "continuous",
  seed = 42,
  label = "",
  minLen = SCHEDULE_DEFAULTS.minLen,
  maxLen = SCHEDULE_DEFAULTS.maxLen,
  duty = SCHEDULE_DEFAULTS.duty,
  cleanPrefix = SCHEDULE_DEFAULTS.cleanPrefix,
}) {
  if (!MODES.includes(mode)) {
    throw new Error('mode must be ' + MODES.join("|") + ', got "' + mode + '"');
  }
  const next = makeRng(hashSeed(seed, label));

  // frames [cleanPrefix, nFrames) are available to the attack
  const avail = Math.max(1, nFrames - cleanPrefix);
  const requestedL = integers(next, minLen, maxLen + 1);
  const L = Math.min(requestedL, avail);

  let spanLen = mode === "continuous" ? L : Math.min(avail, Math.ceil(L / Math.max(duty, 1e-9)));
  spanLen = Math.max(spanLen, L);

  const startLo = Math.min(cleanPrefix, Math.max(0, nFrames - spanLen));
  const startHi = nFrames - spanLen; // inclusive
  const start = startHi > startLo ? integers(next, startLo, startHi + 1) : startLo;

  const spanLocal = [];
  for (let i = start; i < start + spanLen && i < nFrames; i++) spanLocal.push(i);

  let attacked;
  let degenerate = false;
  if (mode === "continuous") {
    attacked = spanLocal.slice();
  } else {
    degenerate = spanLen <= L;
    if (degenerate || spanLocal.length < 3) {
      attacked = spanLocal.slice();
    } else {
      const interior = spanLocal.slice(1, -1);
      const k = L - 2;
      if (k <= 0) {
        attacked = [spanLocal[0], spanLocal[spanLocal.length - 1]].slice(0, L);
      } else {
        // evenly spaced interior picks; rounding can collide, so backfill to k
        const picked = new Set();
        for (let i = 0; i < k; i++) {
          const t = k === 1 ? 0 : (i * (interior.length - 1)) / (k - 1);
          picked.add(Math.round(t));
        }
        if (picked.size < k) {
          for (let cand = 0; cand < interior.length && picked.size < k; cand++) picked.add(cand);
        }
        const idxs = [...picked].sort((a, b) => a - b).slice(0, k);
        attacked = [spanLocal[0], spanLocal[spanLocal.length - 1], ...idxs.map((i) => interior[i])]
          .sort((a, b) => a - b);
      }
    }
  }

  const attackedSet = new Set(attacked);
  const gapLocal = spanLocal.filter((f) => !attackedSet.has(f));

  return {
    attacked,
    isAttacked: (i) => attackedSet.has(i),
    meta: {
      mode,
      nFrames,
      attackLenL: attacked.length,
      requestedL,
      spanStart: start,
      spanLen: spanLocal.length,
      attackedLocal: attacked,
      gapLocal,
      nGaps: gapLocal.length,
      realisedDuty: spanLocal.length ? attacked.length / spanLocal.length : 0,
      degenerate,
      cleanPrefix,
      seed,
    },
  };
}
