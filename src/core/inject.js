/**
 * Occlusion-aware phantom injection.
 *
 * The occlusion core is a faithful port of nuscenes_atlas_common.spoof_frame:
 * per (azimuth x beam) cell, compare the phantom's range against the sorted real
 * returns and let the sensor's return mode decide what survives. Layered on top
 * are the physical gates the offline pipeline could not express, because it
 * assumed a synchronized attacker and worked directly on point clouds:
 *
 *   unreachable  - the cell is outside the spoofer's power envelope, or line of
 *                  sight to the victim is blocked
 *   mot          - the range is inside the sensor's Minimum Operational
 *                  Threshold, so firmware discards the echo
 *   tof          - the range is unreachable in time: nearer than the relay floor
 *                  (needs pulse prediction) or beyond the receiver's range gate
 *   response     - the receiver's angular leakage response at this cell was too
 *                  weak on this shot; strongest at the aim point, falling off
 *                  with angular offset
 *   sync         - the pulse missed its range slot, at a probability set by the
 *                  spoofer's lock quality
 *   cellConflict - another phantom point already claimed this cell; a sensor
 *                  reports at most one return per cell per return slot
 *   occluded     - real geometry lies closer on the same ray
 *
 * Every candidate lands in exactly one of these buckets or is injected, so the
 * counts always reconcile against `requested`.
 *
 * ATLAS-FAITHFUL MODE (`atlasFaithful: true`)
 * ATLAS's spoof_frame applies none of the gates above. It has no power envelope,
 * no timing model and no MOT, and it evaluates each phantom point INDEPENDENTLY
 * with no per-cell dedupe - so several phantom points may share one (az, beam)
 * cell. That is not an oversight to be corrected here: it is exactly why
 * osf_k6nf2/realizability.py exists, to ask afterwards whether the resulting
 * phantom is physically producible (fill <= 1 return per cell). Faithful mode
 * therefore keeps only the occlusion resolution, the ground anchoring and the
 * range-conditioned intensity, and lets fill exceed 1 so the realizability panel
 * can flag it.
 */
import { cartToSpherical, azimuthCellIndex, nearestBeamIndex, cellKey } from "./polar.js";
import { reachableCells } from "./spoofer.js";
import { minRelayRange, syncHitProbability, tofBudget } from "./tof.js";
import { assignIntensity, atlasRangeConditionedIntensity } from "./intensity.js";

/** Deterministic PRNG (mulberry32) so a seed reproduces a whole attack. */
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

/**
 * Vertical shift that puts the phantom's base on the local ground, using the
 * 5th percentile of scene z within `radiusM` of the phantom centroid (ATLAS's
 * ground_anchor_spoof). Returns 0 when there is too little local support.
 */
export function groundAnchorZ({ frame, phantom, radiusM = 2.0, minSupport = 20 }) {
  const n = phantom.x.length;
  if (n === 0 || frame.count === 0) return 0;

  let cx = 0;
  let cy = 0;
  for (let i = 0; i < n; i++) {
    cx += phantom.x[i];
    cy += phantom.y[i];
  }
  cx /= n;
  cy /= n;

  const r2 = radiusM * radiusM;
  const local = [];
  for (let i = 0; i < frame.count; i++) {
    const dx = frame.x[i] - cx;
    const dy = frame.y[i] - cy;
    if (dx * dx + dy * dy < r2) local.push(frame.z[i]);
  }
  if (local.length <= minSupport) return 0;

  local.sort((a, b) => a - b);
  const groundZ = local[Math.floor(0.05 * (local.length - 1))];

  let minZ = Infinity;
  for (let i = 0; i < n; i++) if (phantom.z[i] < minZ) minZ = phantom.z[i];
  return groundZ - minZ;
}

export function injectPhantom({
  frame,
  sensor,
  spoofer,
  positionXYZ,
  phantom,
  reach = null,
  dataset = null,
  intensityMode = "range-conditioned",
  constantIntensity = 0,
  seed = 42,
  groundAnchor = true,
  groundRadiusM = 2.0,
  syncToleranceM = 0.5,
  atlasFaithful = false,
}) {
  const next = makeRng(seed);
  const nReq = phantom.x.length;
  const nBeams = sensor.nBeams;
  const ds = dataset ?? sensor.dataset ?? "nuscenes";

  const rejected = {
    unreachable: 0,
    response: 0,
    tof: 0,
    mot: 0,
    sync: 0,
    cellConflict: 0,
    occluded: 0,
  };

  const reachInfo =
    reach ?? reachableCells({ frame, sensor, spoofer, positionXYZ });

  // cell -> angular response weight, for the sync probability per cell
  const reachWeight = new Map();
  for (let i = 0; i < reachInfo.cells.length; i++) {
    reachWeight.set(reachInfo.cells[i], reachInfo.weight[i]);
  }

  const budget = tofBudget({
    sensor,
    standoffM: reachInfo.geometry.standoff,
    latencyS: spoofer.latencyS,
    predictive: spoofer.canPredict,
  });
  const relayFloor = minRelayRange({
    standoffM: reachInfo.geometry.standoff,
    latencyS: spoofer.latencyS,
  });
  const gateMaxRange = Math.min(sensor.maxRange, sensor.tofToRange(sensor.rangeGate));

  const dz = groundAnchor ? groundAnchorZ({ frame, phantom, radiusM: groundRadiusM }) : 0;

  // ATLAS assigns intensity BEFORE resolving occlusion, to EVERY phantom point,
  // from the single band containing the phantom's MEAN range (r_avg). Doing it
  // afterwards on survivors, or per point, both diverge from that.
  let meanR = 0;
  for (let i = 0; i < nReq; i++) {
    meanR += Math.hypot(phantom.x[i], phantom.y[i], phantom.z[i] + dz);
  }
  meanR = nReq > 0 ? meanR / nReq : 0;
  const intensityAll =
    intensityMode === "range-conditioned"
      ? atlasRangeConditionedIntensity({
          dataset: ds, meanRange: meanR, count: nReq, next,
          fallback: phantom.intensity,
        })
      : assignIntensity({
          mode: intensityMode, dataset: ds, ranges: new Float64Array(nReq),
          next, constantValue: constantIntensity,
        });

  // --- gate every candidate ----------------------------------------------
  const claimed = new Map(); // cell -> {r, x, y, z}  (physics mode)
  const candidates = []; // ordered list            (ATLAS-faithful mode)
  for (let i = 0; i < nReq; i++) {
    const px = phantom.x[i];
    const py = phantom.y[i];
    const pz = phantom.z[i] + dz;
    const { r, az, el } = cartToSpherical(px, py, pz);
    const key = cellKey(azimuthCellIndex(az, sensor.azRes), nearestBeamIndex(el, sensor.inclinations), nBeams);

    if (!atlasFaithful) {
      const weight = reachWeight.get(key);
      if (weight === undefined) {
        rejected.unreachable++;
        continue;
      }
      // Firmware discards sub-MOT echoes regardless of the attacker.
      if (r < sensor.minRange) {
        rejected.mot++;
        continue;
      }
      if (r > gateMaxRange || (!spoofer.canPredict && r < relayFloor)) {
        rejected.tof++;
        continue;
      }
      // Two independent physical effects, kept separate so the readout is
      // diagnostic: the angular leakage response decides whether enough power
      // reaches this cell at all, and lock quality decides whether the pulse
      // lands in its slot. Folding them into one draw would report a fringe cell
      // that never received power as a "sync miss".
      if (next() > weight) {
        rejected.response++;
        continue;
      }
      const pSync = syncHitProbability({
        sensor,
        mode: spoofer.syncMode,
        jitterS: spoofer.jitterS,
        toleranceM: syncToleranceM,
      });
      if (next() > pSync) {
        rejected.sync++;
        continue;
      }

      const prev = claimed.get(key);
      if (prev !== undefined) {
        rejected.cellConflict++;
        // A first/strongest-return sensor reports the nearer echo.
        if (r < prev.r) claimed.set(key, { r, x: px, y: py, z: pz, ii: intensityAll[i] });
        continue;
      }
      claimed.set(key, { r, x: px, y: py, z: pz, ii: intensityAll[i] });
    } else {
      // No dedupe: ATLAS keeps every phantom point independently.
      candidates.push({ key, r, x: px, y: py, z: pz, ii: intensityAll[i] });
    }
  }

  // --- resolve against real returns, per ATLAS spoof_frame ----------------
  const keepMask = new Uint8Array(frame.count).fill(1);
  const keptX = [];
  const keptY = [];
  const keptZ = [];
  const keptR = [];
  const keptI = [];
  let suppressed = 0;

  const resolveList = atlasFaithful
    ? candidates
    : Array.from(claimed, ([key, p]) => ({ key, ...p }));

  for (const p of resolveList) {
    const key = p.key;
    const { start, end } = frame.cellRange(key);
    if (end > start) {
      const nearestIdx = frame.order[start];
      const rNear = frame.r[nearestIdx];
      if (rNear < p.r) {
        // real geometry is in front: the phantom never reaches the receiver
        rejected.occluded++;
        continue;
      }
      // The phantom is the first return. On a single-return sensor everything
      // behind it is destroyed; with dual return the nearest real echo usually
      // survives as the second return.
      const keepSecond = next() < sensor.pTwo;
      const from = keepSecond ? start + 1 : start;
      for (let i = from; i < end; i++) {
        if (keepMask[frame.order[i]]) {
          keepMask[frame.order[i]] = 0;
          suppressed++;
        }
      }
    }
    keptX.push(p.x);
    keptY.push(p.y);
    keptZ.push(p.z);
    keptR.push(p.r);
    keptI.push(p.ii);
  }

  const nInj = keptX.length;
  const spoofR = Float64Array.from(keptR);
  const intensity = Float32Array.from(keptI);

  const spoofAz = new Float64Array(nInj);
  const spoofEl = new Float64Array(nInj);
  const spoofCell = new Int32Array(nInj);
  for (let i = 0; i < nInj; i++) {
    const s = cartToSpherical(keptX[i], keptY[i], keptZ[i]);
    spoofAz[i] = s.az;
    spoofEl[i] = s.el;
    spoofCell[i] = cellKey(
      azimuthCellIndex(s.az, sensor.azRes),
      nearestBeamIndex(s.el, sensor.inclinations),
      nBeams
    );
  }

  return {
    keepMask,
    spoof: {
      x: Float64Array.from(keptX),
      y: Float64Array.from(keptY),
      z: Float64Array.from(keptZ),
      r: spoofR,
      az: spoofAz,
      el: spoofEl,
      cell: spoofCell,
      intensity,
    },
    reach: reachInfo,
    stats: {
      requested: nReq,
      injected: nInj,
      suppressed,
      rejected,
      groundShiftZ: dz,
      cellsReachable: reachInfo.cells.length,
      // In ATLAS-faithful mode there is no envelope, so fill is measured against
      // the cells the phantom itself occupies - and may exceed 1.
      fill: atlasFaithful
        ? (() => {
            const occupied = new Set();
            for (const c of resolveList) occupied.add(c.key);
            return occupied.size > 0 ? nInj / occupied.size : 0;
          })()
        : reachInfo.cells.length > 0
          ? nInj / reachInfo.cells.length
          : 0,
      atlasFaithful,
      relayFloor,
      budget,
      meanPhantomRange: meanR,
      firingLimited: reachInfo.firingLimited,
    },
  };
}
