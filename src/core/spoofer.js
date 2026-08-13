/**
 * Spoofer hardware model and reachable-cell computation.
 *
 * The MVP treated the attack envelope as a free-parameter wedge. It is not: the
 * set of lattice cells an attacker can populate follows from hardware and
 * geometry, and this module derives it.
 *
 * WHY THE ENVELOPE IS WIDER THAN ONE CELL
 * A victim receiver is co-boresighted with its emitter, so a point-like source
 * at bearing b is, to first order, only accepted into the single cell whose
 * instantaneous field of view contains b (~0.17 deg on these sensors). Yet the
 * real capture in Cao et al. shows a 6.8 deg azimuth envelope and all 16/16
 * channels affected. The mechanism is saturation: once received power greatly
 * exceeds the detection threshold, energy leaks into neighbouring detector
 * channels through optical crosstalk and blooming. We therefore model the
 * receiver's angular leakage as a Gaussian response and call a cell reachable
 * when the leaked power still clears the threshold:
 *
 *     response(d_az, d_el) = exp(-d_az^2 / 2 sigma_az^2 - d_el^2 / 2 sigma_el^2)
 *     reachable  <=>  powerRatio * response >= 1
 *     halfWidth  =    sigma * sqrt(2 * ln(powerRatio))
 *
 * CALIBRATION HONESTY
 * sigma_az and the reference powerRatio are *jointly fitted* to two measured
 * anchors from the real capture - the 6.8 deg envelope width and the ~0.30
 * edge/peak per-cell rate ratio (0.25 at the fringe vs 0.83-0.87 at the aim
 * point). Only the combination sigma*sqrt(2 ln R) is identified by the width, so
 * neither constant should be read as an independently measured quantity. What is
 * physically meaningful is the *shape* (Gaussian falloff) and the *scaling*
 * (envelope grows with sqrt(log power), shrinks with distance as 1/d^2).
 *
 * Sources: Cao et al., "You Can't See Me: Physical Removal Attacks on
 * LiDAR-based Autonomous Vehicles Driving Frameworks", USENIX Security 2023
 * (arXiv:2210.09482), as measured in ATLAS_REAL_WORLD_VALIDATION.md.
 */
import { cartToSpherical, azimuthCellIndex, nearestBeamIndex, cellKey, wrapAz, TWO_PI } from "./polar.js";

const DEG = Math.PI / 180;

/** Measured properties of the real VLP-16 attack capture. */
export const CAO2023_ENVELOPE = Object.freeze({
  azFullWidthDeg: 6.8,
  azHalfWidthDeg: 3.4,
  peakRemovalRate: 0.85, // 0.83-0.87 at the aim point
  edgeRemovalRate: 0.25, // at the fringe of the saturated envelope
  targetRangeM: 2.01,
  channelsAffected: 16,
  sensor: "vlp16",
  source: "Cao et al., USENIX Security 2023 (arXiv:2210.09482)",
});

/**
 * Maximum attack angle the paper states as its capability, with end-to-end
 * experiments run at 5 and 10 deg. Wider spans are extrapolations and are
 * flagged as such rather than forbidden.
 */
export const DEMONSTRATED_MAX_AZ_DEG = 45;

/** Fitted receiver angular-leakage widths (see calibration note above). */
export const SIGMA_AZ_DEG = 2.19;
/** Vertical crosstalk is far stronger: the real capture loses all 16 channels. */
export const SIGMA_EL_DEG = 12.0;

export const SPOOFERS = {
  cao2023: {
    name: "Bench relay spoofer (Cao et al. 2023)",
    wavelengthNm: 905,
    apertureM: 0.05,
    divergenceMrad: 3.0,
    latencyS: 10e-9,
    prfHz: 100000,
    canPredict: true,
    syncMode: "locked",
    jitterS: 1e-9,
    powerRatio: 3.33,
    referenceStandoffM: 40,
    note: "Power ratio is quoted at the reference standoff and falls as 1/d^2.",
  },
  lowcost: {
    name: "Low-cost free-running spoofer",
    wavelengthNm: 905,
    apertureM: 0.01,
    divergenceMrad: 10.0,
    latencyS: 150e-9,
    prfHz: 20000,
    canPredict: false,
    syncMode: "free",
    jitterS: 50e-9,
    powerRatio: 1.6,
    referenceStandoffM: 40,
  },
  advanced: {
    name: "High-power synchronized spoofer",
    wavelengthNm: 905,
    apertureM: 0.1,
    divergenceMrad: 1.0,
    latencyS: 2e-9,
    prfHz: 2000000,
    canPredict: true,
    syncMode: "locked",
    jitterS: 100e-12,
    powerRatio: 200,
    referenceStandoffM: 40,
  },
};

export function listSpoofers() {
  return Object.keys(SPOOFERS);
}

export function configureSpoofer(id, overrides = {}) {
  const base = SPOOFERS[id];
  if (!base) {
    throw new Error(
      'unknown spoofer id "' + id + '" (known: ' + listSpoofers().join(", ") + ")"
    );
  }
  const s = { id, ...base, ...overrides };
  if (s.referenceDivergenceMrad == null) s.referenceDivergenceMrad = base.divergenceMrad;
  return Object.freeze(s);
}

/** Standoff, bearing and elevation of the spoofer in the victim sensor frame. */
export function spooferGeometry({ positionXYZ }) {
  const [x, y, z] = positionXYZ;
  const { r, az, el } = cartToSpherical(x, y, z);
  return { standoff: r, bearing: az, elevation: el };
}

/**
 * Received power over detection threshold at the aim point.
 *
 * Falls as 1/d^2 with standoff (spreading loss) and as 1/divergence^2 (a tighter
 * beam concentrates the same pulse energy). Quoted relative to the preset's
 * reference standoff and divergence.
 */
export function effectivePowerRatio({ spoofer, standoff }) {
  const dRef = spoofer.referenceStandoffM;
  const distanceTerm = standoff > 0 ? (dRef / standoff) ** 2 : Infinity;
  const divTerm = (spoofer.referenceDivergenceMrad / spoofer.divergenceMrad) ** 2;
  return spoofer.powerRatio * distanceTerm * divTerm;
}

/**
 * Angular half-widths of the reachable envelope.
 * `standoff` is optional; omit it to evaluate at the quoted reference distance.
 */
export function angularEnvelope({ spoofer, sensor, standoff = null }) {
  const R = standoff == null ? spoofer.powerRatio : effectivePowerRatio({ spoofer, standoff });
  const sigmaAz = SIGMA_AZ_DEG * DEG;
  const sigmaEl = SIGMA_EL_DEG * DEG;

  if (!(R > 1)) {
    return {
      powerRatio: R,
      sigmaAz,
      sigmaEl,
      halfWidthAz: 0,
      halfWidthEl: 0,
      feasible: false,
      extrapolation: false,
    };
  }

  const spread = Math.sqrt(2 * Math.log(R));
  // A cell is always reachable if the spoofer's own bearing lands inside it.
  const ifovFloor = sensor.beamDivergenceMrad / 1000 / 2;
  const halfWidthAz = Math.max(sigmaAz * spread, ifovFloor);
  const halfWidthEl = Math.max(sigmaEl * spread, ifovFloor);

  return {
    powerRatio: R,
    sigmaAz,
    sigmaEl,
    halfWidthAz,
    halfWidthEl,
    feasible: true,
    extrapolation: (2 * halfWidthAz) / DEG > DEMONSTRATED_MAX_AZ_DEG,
  };
}

/**
 * Is there real geometry between the victim and the spoofer?
 *
 * A spoofer parked behind a building cannot inject at any power. Probes the
 * lattice cells around the spoofer bearing and requires agreement across several
 * of them, so one stray return cannot veto an otherwise viable attack.
 */
export function checkLineOfSight({
  frame,
  sensor,
  bearing,
  elevation,
  standoff,
  azProbeCells = 1,
  rowProbe = 4,
  marginM = 0.5,
}) {
  const centreAz = azimuthCellIndex(bearing, sensor.azRes);
  const centreRow = nearestBeamIndex(elevation, sensor.inclinations);

  let occupied = 0;
  let blocked = 0;
  let occluderRange = Infinity;

  for (let da = -azProbeCells; da <= azProbeCells; da++) {
    const a = (((centreAz + da) % sensor.azCellCount) + sensor.azCellCount) % sensor.azCellCount;
    for (let dr = -rowProbe; dr <= rowProbe; dr++) {
      const row = centreRow + dr;
      if (row < 0 || row >= sensor.nBeams) continue;
      const idx = frame.nearestInCell(cellKey(a, row, sensor.nBeams));
      if (idx < 0) continue;
      occupied++;
      const r = frame.r[idx];
      if (r < standoff - marginM) {
        blocked++;
        if (r < occluderRange) occluderRange = r;
      }
    }
  }

  // Require both an absolute and a relative majority: a lone return is noise,
  // but a surface fills most of what the spoofer illuminates.
  const clear = !(blocked >= 2 && blocked >= 0.5 * occupied);
  return {
    clear,
    blocked,
    probed: occupied,
    occluderRange: Number.isFinite(occluderRange) ? occluderRange : null,
  };
}

/**
 * The lattice cells this spoofer can actually populate, with a per-cell response
 * weight in (0, 1] that peaks at the aim point.
 *
 * Constraints applied, in order: victim max range, line of sight, power over
 * threshold, and finally the spoofer's own firing budget (PRF / victim spin
 * rate), which is spent on the highest-weight cells first.
 */
export function reachableCells({ frame, sensor, spoofer, positionXYZ }) {
  const geometry = spooferGeometry({ positionXYZ });
  const { standoff, bearing, elevation } = geometry;
  const maxCellsPerRevolution = Math.floor(spoofer.prfHz / sensor.hz);

  const empty = {
    geometry,
    envelope: angularEnvelope({ spoofer, sensor, standoff }),
    lineOfSight: { clear: true, blocked: 0, probed: 0, occluderRange: null },
    cells: new Int32Array(0),
    weight: new Float32Array(0),
    offsetAz: new Float64Array(0),
    rowOffset: new Int32Array(0),
    maxCellsPerRevolution,
    firingLimited: false,
    feasible: false,
    reason: null,
  };

  if (standoff > sensor.maxRange) {
    return { ...empty, reason: "spoofer stands beyond the victim's maximum range" };
  }

  const lineOfSight = checkLineOfSight({ frame, sensor, bearing, elevation, standoff });
  if (!lineOfSight.clear) {
    return { ...empty, lineOfSight, reason: "line of sight to the victim is occluded" };
  }

  const envelope = angularEnvelope({ spoofer, sensor, standoff });
  if (!envelope.feasible) {
    return {
      ...empty,
      envelope,
      lineOfSight,
      reason: "received power is below the victim's detection threshold",
    };
  }

  const { halfWidthAz, halfWidthEl, sigmaAz, sigmaEl, powerRatio } = envelope;
  const centreRow = nearestBeamIndex(elevation, sensor.inclinations);
  const azSpan = Math.ceil(halfWidthAz / sensor.azRes) + 1;
  const centreAz = azimuthCellIndex(bearing, sensor.azRes);

  const keys = [];
  const weights = [];
  const offAz = [];
  const rowOff = [];

  for (let da = -azSpan; da <= azSpan; da++) {
    const a =
      (((centreAz + da) % sensor.azCellCount) + sensor.azCellCount) % sensor.azCellCount;
    const cellAz = (a + 0.5) * sensor.azRes;
    let dAz = wrapAz(cellAz - bearing);
    if (dAz > Math.PI) dAz -= TWO_PI;
    if (Math.abs(dAz) > halfWidthAz) continue;

    for (let row = 0; row < sensor.nBeams; row++) {
      const dEl = sensor.inclinations[row] - elevation;
      if (Math.abs(dEl) > halfWidthEl) continue;
      const response = Math.exp(
        -(dAz * dAz) / (2 * sigmaAz * sigmaAz) - (dEl * dEl) / (2 * sigmaEl * sigmaEl)
      );
      if (powerRatio * response < 1) continue;
      keys.push(cellKey(a, row, sensor.nBeams));
      weights.push(response);
      offAz.push(dAz);
      rowOff.push(row - centreRow);
    }
  }

  // Spend a limited firing budget on the strongest cells.
  let idx = keys.map((_, i) => i);
  const firingLimited = idx.length > maxCellsPerRevolution;
  if (firingLimited) {
    idx.sort((p, q) => weights[q] - weights[p]);
    idx = idx.slice(0, maxCellsPerRevolution);
  }

  const n = idx.length;
  const cells = new Int32Array(n);
  const weight = new Float32Array(n);
  const offsetAz = new Float64Array(n);
  const rowOffset = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const j = idx[i];
    cells[i] = keys[j];
    weight[i] = weights[j];
    offsetAz[i] = offAz[j];
    rowOffset[i] = rowOff[j];
  }

  return {
    geometry,
    envelope,
    lineOfSight,
    cells,
    weight,
    offsetAz,
    rowOffset,
    maxCellsPerRevolution,
    firingLimited,
    feasible: n > 0,
    reason: n > 0 ? null : "no lattice cell clears the detection threshold",
  };
}
