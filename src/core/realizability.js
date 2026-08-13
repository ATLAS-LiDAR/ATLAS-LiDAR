/**
 * Realizability: is a configured attack inside the physically demonstrated
 * capability envelope?
 *
 * Port of osf_k6nf2/realizability.py, moved from an offline report into live UI
 * feedback. The three quantities an attacker actually cares about:
 *
 *   1. ANGULAR FOOTPRINT - a spoofer can only inject where its beam illuminates
 *      the victim aperture, so a wider span needs more capability.
 *   2. CELL FILL - the hard limit is one spoofed return per (azimuth x beam)
 *      cell, since the attacker cannot create returns in cells the victim never
 *      samples. fill = nPoints / (nAzCells * nBeams); above the sensor's returns
 *      per cell it is impossible, not merely difficult.
 *   3. FIRING BUDGET - points per revolution, bounded by spoofer PRF over victim
 *      spin rate.
 *
 * MEASURED ENVELOPE, AND A CORRECTION
 * Recomputed here from the real VLP-16 captures (real_last.npz, real_dual.npz,
 * n=126 spoofed clusters). The reference script assumes 0.40 deg azimuth
 * resolution for a 10 Hz VLP-16, but the datasheet gives 0.192 deg at 10 Hz
 * (300k points/sec over 16 channels; 0.40 deg corresponds to 20 Hz). The coarse
 * value halves the sampled cell count and therefore doubles measured fill.
 *
 * The decisive evidence that 0.192 deg is correct: at 0.40 deg the measured
 * dual-return fill reaches 2.219 returns per cell, which exceeds the hard limit
 * of 2 for a dual-return sensor. At 0.192 deg it is 1.065 - physically sensible.
 *
 * This matters for interpretation: the published fill envelope is ~2x too
 * permissive, so synthetic phantoms compared against it look more realizable
 * than they are. Both sets of figures are kept below so the comparison is
 * explicit rather than silently corrected.
 */

export const VERDICTS = Object.freeze({
  IMPOSSIBLE: "impossible",
  DEMONSTRATED: "demonstrated",
  WITHIN_CAPABILITY: "within-capability",
  EXTRAPOLATION: "extrapolation",
});

/** Stated maximum attack angle; end-to-end experiments were run at 5 and 10 deg. */
export const DEMONSTRATED_MAX_AZ_DEG = 45;

/**
 * p95 of the real capture, at the datasheet-correct azimuth resolution.
 * Recompute with:
 *   cd osf_k6nf2 && python3 -c "from realizability import real_envelope; ..."
 */
export const DEMONSTRATED_ENVELOPE = Object.freeze({
  sensor: "vlp16",
  hz: 10,
  azResDeg: 0.192,
  azSpanDeg: 7.18,
  beamsSpanned: 16,
  pointsPerRevolution: 586,
  cellFillUnique: 0.283,
  cellFillDual: 1.025,
  nClusters: 126,
  source: "OSF k6nf2 real_last.npz + real_dual.npz (Cao et al., USENIX Sec 2023)",
  asPublished: Object.freeze({
    azResDeg: 0.40,
    azSpanDeg: 7.18,
    cellFillUnique: 0.590,
    cellFillDual: 2.134,
    note: "0.40 deg at 10 Hz is inconsistent with the VLP-16 datasheet; the dual-return fill of 2.134 (max 2.219) exceeds the hard limit of 2 returns per cell.",
  }),
});

/** Port of realizability.cells_and_fill. */
export function cellsAndFill({ azSpanDeg, nBeamsSpanned, nPoints, azResDeg }) {
  const nAz = Math.max(azSpanDeg / azResDeg, 1.0);
  const cells = nAz * Math.max(nBeamsSpanned, 1);
  return { nAz, cells, fill: nPoints / cells };
}

/** Points per revolution the spoofer's own firing rate allows. */
export function firingBudget({ spoofer, sensor, requiredCells = 0 }) {
  const maxCellsPerRevolution = Math.floor(spoofer.prfHz / sensor.hz);
  return {
    maxCellsPerRevolution,
    requiredCells,
    withinBudget: requiredCells <= maxCellsPerRevolution,
    shortfall: Math.max(0, requiredCells - maxCellsPerRevolution),
  };
}

/**
 * Measure the footprint of an actual attack from its reachable cell set.
 * `nInjected` is the number of spoofed returns that survived every gate.
 */
export function analyzeAttack({ reach, sensor, nInjected }) {
  const n = reach.cells.length;
  if (n === 0) {
    return {
      azSpanDeg: 0,
      beamsSpanned: 0,
      nPoints: 0,
      nAz: 0,
      cells: 0,
      fill: 0,
      fillUnique: 0,
      cellsReachable: 0,
    };
  }

  let minOff = Infinity;
  let maxOff = -Infinity;
  const rows = new Set();
  for (let i = 0; i < n; i++) {
    const o = reach.offsetAz[i];
    if (o < minOff) minOff = o;
    if (o > maxOff) maxOff = o;
    rows.add(reach.rowOffset[i]);
  }

  const azSpanDeg = ((maxOff - minOff) * 180) / Math.PI;
  const beamsSpanned = rows.size;
  const azResDeg = (sensor.azRes * 180) / Math.PI;
  const { nAz, cells, fill } = cellsAndFill({
    azSpanDeg,
    nBeamsSpanned: beamsSpanned,
    nPoints: nInjected,
    azResDeg,
  });

  return {
    azSpanDeg,
    beamsSpanned,
    nPoints: nInjected,
    nAz,
    cells,
    fill,
    // Our injection enforces one return per cell, so occupied cells == points.
    fillUnique: cells > 0 ? Math.min(1, nInjected / cells) : 0,
    cellsReachable: n,
    azResDeg,
  };
}

/**
 * Where does this attack sit relative to what has been physically demonstrated?
 * Returns a verdict plus the reasons, so the UI can explain rather than just
 * colour a badge.
 */
export function classifyRealizability(analysis, { maxReturnsPerCell = 1 } = {}) {
  const reasons = [];
  const env = DEMONSTRATED_ENVELOPE;

  if (analysis.fill > maxReturnsPerCell + 1e-9) {
    reasons.push(
      "fill of " + analysis.fill.toFixed(2) +
        " exceeds " + maxReturnsPerCell + " return(s) per cell, which is physically impossible"
    );
    return { verdict: VERDICTS.IMPOSSIBLE, reasons };
  }

  if (analysis.azSpanDeg > DEMONSTRATED_MAX_AZ_DEG) {
    reasons.push(
      "azimuth span of " + analysis.azSpanDeg.toFixed(1) +
        " deg exceeds the stated 45 deg maximum attack angle"
    );
    return { verdict: VERDICTS.EXTRAPOLATION, reasons };
  }

  const wideSpan = analysis.azSpanDeg > env.azSpanDeg;
  const denseFill = analysis.fillUnique > env.cellFillUnique;
  const overBudget = analysis.nPoints > env.pointsPerRevolution;

  if (!wideSpan && !denseFill && !overBudget) {
    reasons.push("inside the measured envelope of the real capture");
    return { verdict: VERDICTS.DEMONSTRATED, reasons };
  }

  if (wideSpan) {
    reasons.push(
      "span " + analysis.azSpanDeg.toFixed(1) + " deg is wider than the measured " +
        env.azSpanDeg + " deg envelope but inside the 45 deg capability"
    );
  }
  if (denseFill) {
    reasons.push(
      "cell fill " + analysis.fillUnique.toFixed(3) + " exceeds the measured " +
        env.cellFillUnique + ", so it is denser than what was demonstrated"
    );
  }
  if (overBudget) {
    reasons.push(
      analysis.nPoints + " points per revolution exceeds the demonstrated " +
        env.pointsPerRevolution
    );
  }
  return { verdict: VERDICTS.WITHIN_CAPABILITY, reasons };
}
