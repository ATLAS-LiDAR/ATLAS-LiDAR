/**
 * Oriented 3D box IoU.
 *
 * The ATLAS success taxonomy is defined entirely through a 3D IoU gate
 * (`boxes_iou3d_gpu(...) >= 0.3` in eval_matrix_nuscenes.eval_asr), so this has
 * to agree with pcdet's convention:
 *
 *   box = [x, y, z, dx, dy, dz, heading]
 *
 * with (x, y, z) the box *centre*, (dx, dy) the footprint aligned to the box's
 * own axes, and `heading` a rotation about +z. pcdet computes 3D IoU as rotated
 * BEV intersection times vertical overlap, which is what we do: exact for boxes
 * that share a vertical axis, which all of these do.
 *
 * The BEV intersection uses Sutherland-Hodgman clipping of one rectangle against
 * the other. Both are convex, so the clip is exact and needs no triangulation.
 */

/** Accept either the pcdet 7-vector or the exporter's object form. */
export function normalizeBox(b) {
  if (Array.isArray(b) || ArrayBuffer.isView(b)) {
    return { center: [b[0], b[1], b[2]], size: [b[3], b[4], b[5]], yaw: b[6] ?? 0 };
  }
  return {
    center: [b.center[0], b.center[1], b.center[2]],
    size: [b.size[0], b.size[1], b.size[2]],
    yaw: b.yaw ?? 0,
    label: b.label,
    score: b.score,
  };
}

/** Footprint corners, counter-clockwise in the box's own frame then rotated. */
export function bevCorners(box) {
  const [x, y] = box.center;
  const hx = box.size[0] / 2;
  const hy = box.size[1] / 2;
  const c = Math.cos(box.yaw);
  const s = Math.sin(box.yaw);
  return [
    [-hx, -hy],
    [hx, -hy],
    [hx, hy],
    [-hx, hy],
  ].map(([lx, ly]) => [x + lx * c - ly * s, y + lx * s + ly * c]);
}

/** Signed shoelace area; sign depends on winding. */
export function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % pts.length];
    a += x1 * y2 - x2 * y1;
  }
  return a / 2;
}

/** Ensure counter-clockwise winding so the inside test has a consistent sign. */
function ccw(pts) {
  return polygonArea(pts) < 0 ? pts.slice().reverse() : pts;
}

/**
 * Sutherland-Hodgman: clip convex polygon `subject` against convex `clip`.
 * Returns [] when they do not overlap.
 */
export function convexIntersection(subject, clip) {
  let out = ccw(subject);
  const cl = ccw(clip);

  for (let i = 0; i < cl.length && out.length > 0; i++) {
    const a = cl[i];
    const b = cl[(i + 1) % cl.length];
    // inside = left of the directed edge a->b
    const side = (p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);

    const input = out;
    out = [];
    for (let j = 0; j < input.length; j++) {
      const cur = input[j];
      const prev = input[(j + input.length - 1) % input.length];
      const dCur = side(cur);
      const dPrev = side(prev);
      const inCur = dCur >= 0;
      const inPrev = dPrev >= 0;

      if (inCur !== inPrev) {
        const t = dPrev / (dPrev - dCur);
        out.push([prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])]);
      }
      if (inCur) out.push(cur);
    }
  }
  return out;
}

/** Rotated bird's-eye-view IoU. */
export function bevIoU(a, b) {
  const ca = bevCorners(a);
  const cb = bevCorners(b);
  const inter = convexIntersection(ca, cb);
  if (inter.length < 3) return 0;
  const ai = Math.abs(polygonArea(inter));
  const aa = Math.abs(polygonArea(ca));
  const ab = Math.abs(polygonArea(cb));
  const union = aa + ab - ai;
  return union > 0 ? ai / union : 0;
}

function zOverlap(a, b) {
  const aLo = a.center[2] - a.size[2] / 2;
  const aHi = a.center[2] + a.size[2] / 2;
  const bLo = b.center[2] - b.size[2] / 2;
  const bHi = b.center[2] + b.size[2] / 2;
  return Math.max(0, Math.min(aHi, bHi) - Math.max(aLo, bLo));
}

/**
 * Is a bird's-eye point inside a box's footprint? Used for click hit-testing so
 * a user can select the vehicle they want erased.
 */
export function pointInBoxBEV(px, py, box) {
  const b = normalizeBox(box);
  const dx = px - b.center[0];
  const dy = py - b.center[1];
  const c = Math.cos(-b.yaw);
  const s = Math.sin(-b.yaw);
  const u = dx * c - dy * s;
  const v = dx * s + dy * c;
  return Math.abs(u) <= b.size[0] / 2 && Math.abs(v) <= b.size[1] / 2;
}

/**
 * Index of the box whose footprint contains the point, preferring the smallest
 * so a pedestrian standing beside a bus is still selectable. -1 if none.
 */
export function pickBoxAt(px, py, boxes) {
  let best = -1;
  let bestArea = Infinity;
  for (let i = 0; i < boxes.length; i++) {
    if (!pointInBoxBEV(px, py, boxes[i])) continue;
    const b = normalizeBox(boxes[i]);
    const area = b.size[0] * b.size[1];
    if (area < bestArea) {
      bestArea = area;
      best = i;
    }
  }
  return best;
}

/** 3D IoU, matching pcdet: rotated BEV intersection x vertical overlap. */
export function iou3d(a, b) {
  const dz = zOverlap(a, b);
  if (dz <= 0) return 0;

  const ca = bevCorners(a);
  const cb = bevCorners(b);
  const inter = convexIntersection(ca, cb);
  if (inter.length < 3) return 0;

  const interArea = Math.abs(polygonArea(inter));
  const interVol = interArea * dz;
  const volA = a.size[0] * a.size[1] * a.size[2];
  const volB = b.size[0] * b.size[1] * b.size[2];
  const union = volA + volB - interVol;
  if (union <= 0) return 0;
  return Math.min(1, Math.max(0, interVol / union));
}

/**
 * Best 3D IoU between a target box and a list of predictions.
 * An empty prediction list yields {iou: 0, index: -1}, which is what makes a
 * removal attack a success and an injection attack a failure.
 */
export function maxIoU(preds, target) {
  const t = normalizeBox(target);
  let best = 0;
  let index = -1;
  for (let i = 0; i < preds.length; i++) {
    const v = iou3d(normalizeBox(preds[i]), t);
    if (v > best) {
      best = v;
      index = i;
    }
  }
  return { iou: best, index };
}
