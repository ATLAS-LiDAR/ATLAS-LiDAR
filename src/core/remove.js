/**
 * Physical Removal Attack (PRA): destroying genuine returns by injection.
 *
 * The MVP simply deleted every point inside a wedge. That is an assertion, not a
 * mechanism. The real attack, from Cao et al. (USENIX Security 2023), works like
 * this:
 *
 *   1. The attacker fires a *saturated* echo on the victim's ray, timed to
 *      appear at a range inside the sensor's Minimum Operational Threshold
 *      (40 cm on the VLP-16).
 *   2. The victim prefers the strongest echo, so the attacker's pulse wins over
 *      the genuine surface return.
 *   3. Firmware discards sub-MOT returns - so the winning echo is thrown away
 *      too, and the ray reports nothing at all.
 *
 * Removal is therefore achieved *via* injection, and it follows that the attack
 * needs pulse prediction: a sub-MOT echo has to arrive within ~2.7 ns of the
 * victim firing, which a spoofer metres away cannot do reactively. This module
 * derives that constraint rather than assuming a synchronized attacker.
 *
 * Three outcomes fall out of where the attacker's echo lands:
 *
 *   attackerRange <  MOT            -> REMOVED  (return disappears)
 *   MOT <= attackerRange < genuine  -> REPLACED (return moves; this is what the
 *                                      pre-MOT-filter pcaps actually show, a
 *                                      2.01 m target reappearing at ~11 m)
 *   attackerRange >= genuine        -> OCCLUDED (attack has no effect)
 *
 * Per-cell success probability is peak rate x angular response x sync quality,
 * which reproduces the measured azimuth-peaked profile: 0.83-0.87 at the aim
 * point falling to ~0.25 at the fringe of the saturated envelope. A flat rate
 * gets this wrong by ~2x when averaged over the envelope.
 */
import { reachableCells, CAO2023_ENVELOPE } from "./spoofer.js";
import { syncHitProbability } from "./tof.js";
import { sphericalToCart } from "./polar.js";

export const OUTCOMES = Object.freeze({
  REMOVED: "removed",
  REPLACED: "replaced",
  OCCLUDED: "occluded",
  INFEASIBLE: "infeasible",
});

function makeRng(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function removeReturns({
  frame,
  sensor,
  spoofer,
  positionXYZ,
  attackerRangeM = 0.2,
  reach = null,
  seed = 42,
  peakRate = CAO2023_ENVELOPE.peakRemovalRate,
  syncToleranceM = 0.5,
  saturatedIntensity = null,
}) {
  const next = makeRng(seed);
  const keepMask = new Uint8Array(frame.count).fill(1);
  const reachInfo = reach ?? reachableCells({ frame, sensor, spoofer, positionXYZ });

  const empty = (reason) => ({
    keepMask,
    replacement: { x: new Float64Array(0), y: new Float64Array(0), z: new Float64Array(0), r: new Float64Array(0), intensity: new Float32Array(0) },
    reach: reachInfo,
    stats: {
      outcome: OUTCOMES.INFEASIBLE,
      reason,
      removed: 0,
      replaced: 0,
      envelopeReturns: 0,
      envelopeMeanRate: 0,
      peakRate: 0,
      edgeRate: 0,
      affectedCells: new Int32Array(0),
    },
  });

  if (!reachInfo.feasible) {
    return empty(reachInfo.reason ?? "spoofer cannot reach the victim");
  }

  // A sub-MOT echo must arrive essentially at the moment of firing, which is
  // only possible if the attacker predicts the next pulse.
  const subMot = attackerRangeM < sensor.minRange;
  if (subMot && !spoofer.canPredict) {
    return empty(
      "a sub-MOT echo requires firing before the victim's pulse arrives, i.e. pulse prediction"
    );
  }

  const pSync = syncHitProbability({
    sensor,
    mode: spoofer.syncMode,
    jitterS: spoofer.jitterS,
    toleranceM: syncToleranceM,
  });

  const halfAz = reachInfo.envelope.halfWidthAz;
  const affected = [];
  const repX = [];
  const repY = [];
  const repZ = [];
  const repR = [];

  let removed = 0;
  let replaced = 0;
  let envelopeReturns = 0;
  let occludedCells = 0;

  // Rate profile bins, for reporting the azimuth peaking explicitly.
  //
  // Both bands are restricted to the central beam rows, because the measured
  // figures they are compared against are per-aim-point, not per-envelope: the
  // real capture reports 0.83-0.87 "at the aim point / best-aimed frames" and
  // ~0.25 at the fringe. Averaging over all rows instead would dilute the peak
  // by the vertical response falloff (0.46 at the edge of a 30 deg VFOV) and
  // understate it by ~20%, which is the same averaging mistake the validation
  // doc flags for the envelope mean.
  let peakHit = 0;
  let peakTotal = 0;
  let edgeHit = 0;
  let edgeTotal = 0;
  const peakBand = Math.max(0.6 * Math.PI / 180, sensor.azRes * 3);
  const edgeBand = 0.85 * halfAz;
  const centreRows = 2;

  for (let c = 0; c < reachInfo.cells.length; c++) {
    const key = reachInfo.cells[c];
    const { start, end } = frame.cellRange(key);
    if (end === start) continue;

    const nReturns = end - start;
    envelopeReturns += nReturns;

    const offset = Math.abs(reachInfo.offsetAz[c]);
    const nearCentreRow = Math.abs(reachInfo.rowOffset[c]) <= centreRows;
    const inPeak = nearCentreRow && offset <= peakBand;
    const inEdge = nearCentreRow && offset >= edgeBand;
    if (inPeak) peakTotal += nReturns;
    if (inEdge) edgeTotal += nReturns;

    // The genuine return the attacker is competing with.
    const nearest = frame.order[start];
    if (attackerRangeM >= frame.r[nearest]) {
      occludedCells++;
      continue;
    }

    const p = peakRate * reachInfo.weight[c] * pSync;
    if (next() >= p) continue;

    affected.push(key);
    for (let i = start; i < end; i++) {
      const pt = frame.order[i];
      if (!keepMask[pt]) continue;
      keepMask[pt] = 0;
      if (subMot) {
        removed++;
        if (inPeak) peakHit++;
        if (inEdge) edgeHit++;
      } else {
        // Above the MOT the echo survives, so the return is not deleted but
        // range-replaced along the same ray.
        replaced++;
        if (inPeak) peakHit++;
        if (inEdge) edgeHit++;
        const [rx, ry, rz] = sphericalToCart(attackerRangeM, frame.az[pt], frame.el[pt]);
        repX.push(rx);
        repY.push(ry);
        repZ.push(rz);
        repR.push(attackerRangeM);
      }
    }
  }

  let outcome;
  if (removed > 0) outcome = OUTCOMES.REMOVED;
  else if (replaced > 0) outcome = OUTCOMES.REPLACED;
  else if (occludedCells > 0) outcome = OUTCOMES.OCCLUDED;
  else outcome = OUTCOMES.INFEASIBLE;

  const affectedTotal = removed + replaced;
  const fullScale = saturatedIntensity ?? sensor.intensityScaleMax;
  const intensity = new Float32Array(repR.length).fill(fullScale);

  return {
    keepMask,
    replacement: {
      x: Float64Array.from(repX),
      y: Float64Array.from(repY),
      z: Float64Array.from(repZ),
      r: Float64Array.from(repR),
      intensity,
    },
    reach: reachInfo,
    stats: {
      outcome,
      reason: outcome === OUTCOMES.OCCLUDED ? "attacker echo lies behind the genuine return" : null,
      removed,
      replaced,
      envelopeReturns,
      envelopeMeanRate: envelopeReturns > 0 ? affectedTotal / envelopeReturns : 0,
      peakRate: peakTotal > 0 ? peakHit / peakTotal : 0,
      edgeRate: edgeTotal > 0 ? edgeHit / edgeTotal : 0,
      affectedCells: Int32Array.from(affected),
      subMot,
    },
  };
}
