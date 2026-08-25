/**
 * Where the attacker's hardware stands, and whether the attack it is being asked
 * to produce is reachable from there.
 *
 * TWO PLACEMENT MODES
 *   static  - a roadside box. It holds ONE world position for the whole sequence,
 *             so its ego-relative bearing and range both change every frame: it
 *             approaches, is passed, and recedes. That apparent motion is the ego
 *             moving, not the attacker moving.
 *   dynamic - a vehicle-mounted or chase attacker. It holds a fixed offset from
 *             the ego, so it travels along and its geometry never changes.
 *
 * An ego-fixed phantom (place_trace_relative, always 15-20 m ahead) implicitly
 * requires the dynamic case: a phantom that follows the victim cannot be produced
 * from a box bolted to the roadside for more than a moment.
 *
 * ALIGNMENT IS NOT THE CRITERION
 * An attacker does not have to sit on the azimuth it wants to inject at. Its beam
 * saturates a band around its own bearing - a measured 6.8 deg full width at the
 * reference power in Cao et al., widening as sqrt(log power) and quoted up to a
 * demonstrated 45 deg maximum attack angle. So the question is CONTAINMENT: does
 * the attacked sector fall inside that band? `sectorCoverage` answers it as a
 * fraction, so drift degrades gradually instead of flipping a boolean.
 *
 * Note the resulting dynamic for a roadside attacker: received power goes as
 * 1/d^2, so its reachable band widens as the ego closes and narrows as it
 * recedes, while the bearing offset grows throughout - a window of opportunity
 * that opens and shuts.
 *
 * NONE OF THIS GATES ATLAS. ATLAS has no power model, so in faithful mode the
 * coverage number is a *report* on whether real hardware standing there could
 * have produced the attack. It only becomes a constraint with the physics overlay
 * on.
 */
import { invertRigid4, applyPose } from "./pose.js";
import { angularEnvelope } from "./spoofer.js";
import { wrapAz, TWO_PI } from "./polar.js";

export const ATTACKER_MODES = ["static", "dynamic"];

/**
 * The victim's instantaneous azimuthal acceptance.
 *
 * A receiver co-boresighted with its emitter only accepts light arriving within a
 * narrow cone around the direction the beam currently points - a few milliradians
 * on these sensors, i.e. a fraction of a degree, not degrees. This is the keyhole
 * an attacker has to shoot through.
 */
export function victimApertureRad(sensor) {
  return sensor.beamDivergenceMrad / 1000;
}

/**
 * The arc over which the attacker's light can reach the receiver at all: its
 * reachable band, widened by half the victim's aperture at each end.
 */
export function illuminationWindow({ attackerBearing, envelopeHalfWidthRad, apertureRad }) {
  return {
    centerAz: attackerBearing,
    halfWidthRad: envelopeHalfWidthRad + apertureRad / 2,
  };
}

/**
 * Can the attacker's light get into the receiver right now?
 *
 * Keyed off the ATTACKER's own bearing, not the attacked sector's. Those are the
 * same only when the attacker happens to sit on the sector; a roadside attacker
 * drifts off it as the ego drives past, and firing has to follow where the
 * hardware actually stands.
 *
 * Note a real attacker may pulse continuously at its PRF - what this predicate
 * decides is whether the victim can HEAR it, which is what matters.
 */
export function attackerCanIlluminate({ sweepAz, attackerBearing, envelopeHalfWidthRad, apertureRad }) {
  const w = illuminationWindow({ attackerBearing, envelopeHalfWidthRad, apertureRad });
  let d = wrapAz(sweepAz - w.centerAz);
  if (d > Math.PI) d -= TWO_PI;
  return Math.abs(d) <= w.halfWidthRad;
}

/** Fraction of each revolution during which the attacker can be heard. */
export function illuminationDuty({ envelopeHalfWidthRad, apertureRad }) {
  return Math.min(1, (2 * envelopeHalfWidthRad + apertureRad) / TWO_PI);
}

const DEG = Math.PI / 180;

/** Lift an ego-frame point into world coordinates (for click-to-place). */
export function worldFromEgoPoint(egoPoint, pose) {
  return pose ? applyPose(pose, egoPoint) : egoPoint.slice();
}

/**
 * The attacker's position in the CURRENT frame's ego coordinates.
 *
 * A static attacker needs a pose to project its world position down; without one
 * (a single static frame, or before a position has been chosen) it falls back to
 * the ego offset, where static and dynamic are indistinguishable anyway.
 */
export function attackerEgoPosition({ mode = "static", worldPos = null, egoOffset = [18, 0, 0], pose = null }) {
  if (mode === "dynamic" || !worldPos || !pose) return egoOffset.slice();
  return applyPose(invertRigid4(pose), worldPos);
}

/**
 * How much of the attacked sector falls inside the attacker's reachable band.
 *
 * @returns {{fraction, offsetRad, offsetDeg, withinEnvelope, partial}}
 */
export function sectorCoverage({
  attackerBearing,
  sectorCenterAz,
  sectorWidthRad,
  envelopeHalfWidthRad,
}) {
  let d = wrapAz(sectorCenterAz - attackerBearing);
  if (d > Math.PI) d -= TWO_PI;

  const half = sectorWidthRad / 2;
  const lo = d - half;
  const hi = d + half;
  const reachLo = -envelopeHalfWidthRad;
  const reachHi = envelopeHalfWidthRad;

  const overlap = Math.max(0, Math.min(hi, reachHi) - Math.max(lo, reachLo));
  const fraction = sectorWidthRad > 0 ? Math.min(1, overlap / sectorWidthRad) : 0;

  return {
    fraction,
    offsetRad: d,
    offsetDeg: (d * 180) / Math.PI,
    withinEnvelope: fraction >= 1 - 1e-9,
    partial: fraction > 1e-9 && fraction < 1 - 1e-9,
  };
}

/**
 * Everything the UI needs about the attacker this frame: where it is, how wide
 * its reachable band is at that standoff, and how much of the attacked sector it
 * covers.
 */
export function attackerGeometry({
  mode = "static",
  worldPos = null,
  egoOffset = [18, 0, 0],
  pose = null,
  sensor,
  spoofer,
  sectorCenterAz = 0,
  sectorWidthRad = 0,
}) {
  const egoPos = attackerEgoPosition({ mode, worldPos, egoOffset, pose });
  const standoff = Math.hypot(egoPos[0], egoPos[1], egoPos[2] ?? 0);
  const bearing = Math.atan2(egoPos[1], egoPos[0]);

  const envelope = angularEnvelope({ spoofer, sensor, standoff });
  const coverage = sectorCoverage({
    attackerBearing: bearing,
    sectorCenterAz,
    sectorWidthRad,
    envelopeHalfWidthRad: envelope.feasible ? envelope.halfWidthAz : 0,
  });

  return {
    mode,
    egoPos,
    standoff,
    bearing,
    bearingDeg: (bearing * 180) / Math.PI,
    envelope,
    envelopeHalfWidthDeg: envelope.feasible ? envelope.halfWidthAz / DEG : 0,
    coverage,
  };
}
