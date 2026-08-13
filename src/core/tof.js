/**
 * Time-of-flight and attacker/victim synchronization.
 *
 * ATLAS_REAL_WORLD_VALIDATION.md lists this as explicitly *not modelled*: "laser
 * wavelength, analog receiver electronics, attacker-victim timing
 * synchronization. ATLAS assumes a synchronized attacker and simulates the
 * resulting point-cloud perturbation." This module removes that assumption and
 * makes it a parameter, which is the main reason the sandbox can be more
 * physically honest than the offline pipeline.
 *
 * The chain for a reactive (relay) spoofer standing `d` metres from the victim:
 *
 *   victim fires  ->  d/c  ->  spoofer detects  ->  t_latency  ->  spoofer fires
 *                 ->  d/c  ->  victim receives
 *
 * so the victim measures ToF = 2d/c + t_latency + t_delay and reports
 *
 *   range = c*ToF/2 = d + c*(t_latency + t_delay)/2
 *
 * Two consequences that a point-cloud-level attack model cannot express:
 *
 *  1. A relay spoofer CANNOT place a phantom closer than its own standoff
 *     distance. Doing so needs a negative delay, i.e. firing before the victim's
 *     pulse arrives, which requires predicting the next firing time.
 *  2. Timing precision IS range precision, at c/2 = 0.15 m per nanosecond.
 */
import { C_LIGHT } from "./sensor.js";

/** Metres of apparent range per second of delay. */
export const HALF_C = C_LIGHT / 2;

/** Abramowitz & Stegun 7.1.26; abs error < 1.5e-7, saturates cleanly. */
function erf(x) {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  if (a > 6) return s;
  const t = 1 / (1 + 0.3275911 * a);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-a * a);
  return s * y;
}

/** Range the victim reports for a spoofed echo. */
export function apparentRange({ standoffM, latencyS = 0, delayS = 0 }) {
  return standoffM + (latencyS + delayS) * HALF_C;
}

/**
 * Delay needed to place a phantom at `targetRangeM`. A negative result means the
 * spoofer must fire before the victim's pulse reaches it - only possible with
 * pulse prediction.
 */
export function delayForRange({ targetRangeM, standoffM, latencyS = 0 }) {
  return (2 * (targetRangeM - standoffM)) / C_LIGHT - latencyS;
}

/** Closest range a purely reactive spoofer can produce. */
export function minRelayRange({ standoffM, latencyS = 0 }) {
  return standoffM + latencyS * HALF_C;
}

export function requiresPrediction(targetRangeM, { standoffM, latencyS = 0 }) {
  return targetRangeM < minRelayRange({ standoffM, latencyS });
}

/** Range standard deviation implied by a timing standard deviation. */
export function rangeErrorStd(jitterS) {
  return jitterS * HALF_C;
}

/**
 * Fraction of time the victim's receiver is actually listening. An
 * unsynchronized spoofer firing at random phase is only heard this often - about
 * 2% on a VLP-16, which is why practical attacks all synchronize to the victim.
 */
export function gateDutyCycle(sensor) {
  return sensor.rangeGate / sensor.firingPeriod;
}

/**
 * Probability a spoofed pulse lands close enough to its intended slot to be
 * reported at the intended range.
 *
 * mode "free":   no lock; the pulse hits the open gate at the duty cycle.
 * mode "locked": timing error ~ N(0, jitter); usable when the induced range
 *                error stays inside `toleranceM`. Floored at the free-running
 *                duty cycle, since a badly locked spoofer is never *worse* than
 *                an unsynchronized one.
 */
export function syncHitProbability({ sensor, mode = "locked", jitterS = 0, toleranceM = 0.5 }) {
  const duty = gateDutyCycle(sensor);
  if (mode === "free") return duty;
  if (jitterS <= 0) return 1;
  const tolS = (2 * toleranceM) / C_LIGHT;
  const p = erf(tolS / (jitterS * Math.SQRT2));
  return Math.min(1, Math.max(p, duty));
}

/**
 * The range interval a given spoofer configuration can actually populate.
 * `feasible` is false when the interval is empty - e.g. the spoofer stands
 * farther away than the victim's maximum range.
 */
export function tofBudget({ sensor, standoffM, latencyS = 0, predictive = false }) {
  const relayFloor = minRelayRange({ standoffM, latencyS });
  const minRange = predictive ? sensor.minRange : relayFloor;
  const gateRange = sensor.tofToRange(sensor.rangeGate);
  const maxRange = Math.min(sensor.maxRange, gateRange);
  const maxDelayS = delayForRange({ targetRangeM: maxRange, standoffM, latencyS });
  return {
    minRange,
    maxRange,
    maxDelayS,
    relayFloor,
    requiresPrediction: predictive && relayFloor > sensor.minRange,
    feasible: maxRange > minRange,
  };
}

/** Absolute time within a revolution at which the victim samples an azimuth cell. */
export function predictedFireTime(sensor, azIdx, phaseS = 0) {
  const t = sensor.timeOfAzCell(azIdx) + phaseS;
  const T = sensor.revolutionPeriod;
  const m = t % T;
  return m < 0 ? m + T : m;
}
