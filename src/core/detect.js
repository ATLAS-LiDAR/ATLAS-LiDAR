/**
 * Perception PROXY.
 *
 * ==========================================================================
 *  THIS IS NOT A TRAINED 3D DETECTOR. It is ground removal + BEV connected
 *  components + oriented box fitting. It measures whether an object is
 *  *geometrically* there to be found, which is enough to exercise the ATLAS
 *  success taxonomy interactively, and it reproduces the taxonomy's structure
 *  faithfully.
 *
 *  It does NOT reproduce the vulnerability of PointPillars, VoxelNeXt or
 *  TransFusion. ASR values computed from it are proxy values and must never be
 *  reported as the paper's. For real numbers, export the perturbed frame
 *  (Phase 3) and run it through the detectors in Polar-Hierarchical-Mamba.
 * ==========================================================================
 *
 * Confidence is grounded in the sensor lattice rather than invented: an object
 * of a given size at a given range subtends a computable number of
 * (azimuth x beam) cells, so the ratio of observed points to that count is a
 * physically meaningful "how well sampled is this thing" score.
 */
import { nearestBeamIndex } from "./polar.js";

/** ATLAS class groups (atlas_injection_eval_only groups hits this way). */
export const CLASS_GROUPS = ["vehicle", "pedestrian", "cyclist"];

export const DEFAULTS = Object.freeze({
  groundCellM: 4.0, // coarse cell for local ground estimation
  groundMarginM: 0.3, // keep points this far above local ground
  clusterCellM: 0.3, // BEV occupancy cell; 0.3 measured best (bench.html ablation)
  minPoints: 12,
  // Point count per object falls as 1/r^2, so a fixed floor makes the detector
  // range-blind: it either misses everything far away or fragments everything
  // near. Scale the requirement with the cells the object actually subtends.
  minPointsAtRangeM: 25, // range at which minPoints applies; scaled as (ref/r)^2
  minPointsFloor: 5,
  // Extending an under-sized vehicle box away from the sensor to a class prior
  // is standard practice for partial views, but it MEASURES WORSE here (vehicle
  // recall@0.3 0.24 with vs 0.32 without - see the bench.html ablation), so it
  // ships off. Likely because KITTI clusters often merge a car with adjacent
  // structure, and growing an already-too-large box compounds the error.
  vehiclePrior: false,
  vehicleMinLengthM: 3.9,
  vehicleMinWidthM: 1.6,
  minExtentM: 0.3,
  maxExtentM: 20.0,
  maxHeightM: 4.5,
  fillForFullScore: 0.35, // observed/expected cell fill that scores 1.0
  maxRangeM: 80,
});

/** Andrew's monotone chain convex hull of 2D points, counter-clockwise. */
export function convexHull(pts) {
  if (pts.length < 3) return pts.slice();
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

/**
 * Minimum-area enclosing rectangle via rotating calipers.
 *
 * This replaces a PCA fit, which is the wrong tool here: a LiDAR sees only the
 * one or two faces of a vehicle pointing at it, so the point distribution is
 * L-shaped and its principal axis is a diagonal rather than the vehicle's
 * heading. The min-area rectangle over the hull recovers the heading from the
 * faces themselves.
 */
export function minAreaRect(pts) {
  const hull = convexHull(pts);
  if (hull.length < 3) {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const [x, y] of pts) {
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    return { cx: (xMin + xMax) / 2, cy: (yMin + yMax) / 2, du: xMax - xMin, dv: yMax - yMin, yaw: 0 };
  }

  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const c = Math.cos(-ang);
    const s = Math.sin(-ang);
    let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
    for (const [x, y] of hull) {
      const u = x * c - y * s;
      const v = x * s + y * c;
      if (u < uMin) uMin = u;
      if (u > uMax) uMax = u;
      if (v < vMin) vMin = v;
      if (v > vMax) vMax = v;
    }
    const area = (uMax - uMin) * (vMax - vMin);
    if (!best || area < best.area) {
      const cu = (uMin + uMax) / 2;
      const cv = (vMin + vMax) / 2;
      best = {
        area,
        yaw: ang,
        du: uMax - uMin,
        dv: vMax - vMin,
        // rotate the rect centre back into world axes
        cx: cu * Math.cos(ang) - cv * Math.sin(ang),
        cy: cu * Math.sin(ang) + cv * Math.cos(ang),
      };
    }
  }
  return best;
}

/** Size-based class assignment, matching the ATLAS group granularity. */
export function classifyBySize(size) {
  const [dx, dy, dz] = size;
  const long = Math.max(dx, dy);
  const short = Math.min(dx, dy);
  if (long >= 3.0) return "vehicle";
  if (long >= 1.2 && short < 1.2 && dz >= 1.2) return "cyclist";
  if (long < 1.2 && dz >= 1.2) return "pedestrian";
  return long >= 2.0 ? "vehicle" : "pedestrian";
}

/**
 * Number of lattice cells an object of this footprint and height subtends at
 * this range - the upper bound on how many points it could possibly return.
 */
function expectedCells(sensor, box) {
  const r = Math.hypot(box.center[0], box.center[1]);
  if (r < 1e-6) return 1;
  const width = Math.max(box.size[0], box.size[1]);
  const azSpan = 2 * Math.atan(width / 2 / r);
  const nAz = Math.max(1, azSpan / sensor.azRes);

  const zLo = box.center[2] - box.size[2] / 2;
  const zHi = box.center[2] + box.size[2] / 2;
  const elLo = Math.atan2(zLo, r);
  const elHi = Math.atan2(zHi, r);
  const rowLo = nearestBeamIndex(elLo, sensor.inclinations);
  const rowHi = nearestBeamIndex(elHi, sensor.inclinations);
  const nRows = Math.max(1, Math.abs(rowHi - rowLo) + 1);

  return nAz * nRows;
}

/** Local ground height per coarse BEV cell, as the 5th percentile of z. */
function groundGrid(cloud, cellM, extentM) {
  const nx = Math.ceil((2 * extentM) / cellM);
  const buckets = new Map();
  for (let i = 0; i < cloud.n; i++) {
    const gx = Math.floor((cloud.x[i] + extentM) / cellM);
    const gy = Math.floor((cloud.y[i] + extentM) / cellM);
    if (gx < 0 || gy < 0 || gx >= nx || gy >= nx) continue;
    const key = gy * nx + gx;
    let arr = buckets.get(key);
    if (!arr) {
      arr = [];
      buckets.set(key, arr);
    }
    arr.push(cloud.z[i]);
  }
  const ground = new Map();
  for (const [key, zs] of buckets) {
    zs.sort((a, b) => a - b);
    ground.set(key, zs[Math.floor(0.05 * (zs.length - 1))]);
  }
  return { ground, nx, cellM, extentM };
}

/**
 * Detect objects in a composed cloud.
 * @returns {Array<{center, size, yaw, score, label, nPoints}>}
 */
export function detect(cloud, { sensor, params = {} } = {}) {
  const P = { ...DEFAULTS, ...params };
  if (cloud.n === 0) return [];

  const extent = P.maxRangeM;
  const g = groundGrid(cloud, P.groundCellM, extent);

  // --- keep only above-ground points inside range ---
  const idx = [];
  for (let i = 0; i < cloud.n; i++) {
    const px = cloud.x[i];
    const py = cloud.y[i];
    if (Math.hypot(px, py) > extent) continue;
    const gx = Math.floor((px + extent) / P.groundCellM);
    const gy = Math.floor((py + extent) / P.groundCellM);
    const gz = g.ground.get(gy * g.nx + gx);
    if (gz === undefined) continue;
    if (cloud.z[i] < gz + P.groundMarginM) continue;
    if (cloud.z[i] > gz + P.maxHeightM) continue;
    idx.push(i);
  }
  if (idx.length === 0) return [];

  // --- BEV occupancy grid + connected components ---
  const cell = P.clusterCellM;
  const nx = Math.ceil((2 * extent) / cell);
  const cellOf = (i) => {
    const cx = Math.floor((cloud.x[i] + extent) / cell);
    const cy = Math.floor((cloud.y[i] + extent) / cell);
    return cy * nx + cx;
  };
  const occupancy = new Map(); // cellKey -> point indices
  for (const i of idx) {
    const k = cellOf(i);
    let arr = occupancy.get(k);
    if (!arr) {
      arr = [];
      occupancy.set(k, arr);
    }
    arr.push(i);
  }

  const visited = new Set();
  const clusters = [];
  for (const start of occupancy.keys()) {
    if (visited.has(start)) continue;
    visited.add(start);
    const stack = [start];
    const members = [];
    while (stack.length) {
      const k = stack.pop();
      const pts = occupancy.get(k);
      if (pts) for (const p of pts) members.push(p);
      const cy = Math.floor(k / nx);
      const cx = k - cy * nx;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const ax = cx + dx;
          const ay = cy + dy;
          if (ax < 0 || ay < 0 || ax >= nx || ay >= nx) continue;
          const nk = ay * nx + ax;
          if (visited.has(nk) || !occupancy.has(nk)) continue;
          visited.add(nk);
          stack.push(nk);
        }
      }
    }
    if (members.length < P.minPointsFloor) continue;
    // range-adaptive point requirement, evaluated at the cluster centroid
    let mx = 0, my = 0;
    for (const p2 of members) { mx += cloud.x[p2]; my += cloud.y[p2]; }
    const rC = Math.hypot(mx / members.length, my / members.length) || 1;
    const need = Math.max(
      P.minPointsFloor,
      P.minPoints * Math.min(1, (P.minPointsAtRangeM / rC) ** 2)
    );
    if (members.length >= need) clusters.push(members);
  }

  // --- fit an oriented box per cluster ---
  const dets = [];
  for (const members of clusters) {
    let cx = 0;
    let cy = 0;
    let zMin = Infinity;
    let zMax = -Infinity;
    for (const i of members) {
      cx += cloud.x[i];
      cy += cloud.y[i];
      if (cloud.z[i] < zMin) zMin = cloud.z[i];
      if (cloud.z[i] > zMax) zMax = cloud.z[i];
    }
    cx /= members.length;
    cy /= members.length;

    const rect = minAreaRect(members.map((i) => [cloud.x[i], cloud.y[i]]));
    let { yaw, du, dv } = rect;
    du = Math.max(du, P.minExtentM);
    dv = Math.max(dv, P.minExtentM);
    const dh = Math.max(zMax - zMin, P.minExtentM);
    if (du > P.maxExtentM || dv > P.maxExtentM) continue;

    let center = [rect.cx, rect.cy, (zMin + zMax) / 2];
    let size = [du, dv, dh];

    // A LiDAR only sees the faces pointing at it, so a vehicle's far side is
    // missing and the fitted depth is far too short. Extend an under-sized
    // vehicle footprint AWAY from the sensor to a class prior, keeping the
    // observed near face fixed - the standard correction for partial views.
    if (P.vehiclePrior && classifyBySize(size) === "vehicle") {
      const r = Math.hypot(center[0], center[1]) || 1;
      const radial = [center[0] / r, center[1] / r];
      // the box's own axes in world coordinates
      const axU = [Math.cos(yaw), Math.sin(yaw)];
      const axV = [-Math.sin(yaw), Math.cos(yaw)];
      const dotU = axU[0] * radial[0] + axU[1] * radial[1];
      const dotV = axV[0] * radial[0] + axV[1] * radial[1];

      // Grow along whichever box axis points most nearly away from the sensor,
      // and shift the centre along THAT axis - not along the radial direction,
      // or the box is displaced sideways as well as enlarged.
      const useU = Math.abs(dotU) >= Math.abs(dotV);
      const ai = useU ? 0 : 1;
      const axis = useU ? axU : axV;
      const sign = (useU ? dotU : dotV) >= 0 ? 1 : -1;

      if (size[ai] < P.vehicleMinLengthM) {
        const grow = P.vehicleMinLengthM - size[ai];
        size[ai] = P.vehicleMinLengthM;
        center = [
          center[0] + axis[0] * sign * (grow / 2),
          center[1] + axis[1] * sign * (grow / 2),
          center[2],
        ];
      }
      const other = 1 - ai;
      if (size[other] < P.vehicleMinWidthM) size[other] = P.vehicleMinWidthM;
    }
    const box = { center, size, yaw };
    const expected = expectedCells(sensor, box);
    const fill = members.length / expected;
    const score = Math.min(1, Math.max(0, fill / P.fillForFullScore));

    dets.push({
      center,
      size,
      yaw,
      score,
      label: classifyBySize(size),
      nPoints: members.length,
      expectedCells: expected,
      cellFill: fill,
    });
  }

  // stable, highest-confidence first
  dets.sort((a, b) => b.score - a.score || b.nPoints - a.nPoints || a.center[0] - b.center[0]);
  return dets;
}
