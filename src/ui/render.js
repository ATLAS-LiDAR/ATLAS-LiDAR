/**
 * Rendering.
 *
 * Two deliberate choices, both fixing problems in the single-file MVP:
 *
 *  1. The palette is a hoisted constant. The MVP called getComputedStyle()
 *     *per point* inside the draw loop, forcing a style recalculation thousands
 *     of times per frame.
 *  2. Points are splatted into an ImageData buffer rather than drawn as
 *     individual arc() paths. At 120k points per frame that is the difference
 *     between an interactive view and a slideshow.
 *
 * The range-image view (azimuth x beam) is the one that makes the physics
 * legible: it shows the scan lattice directly, so the reachable envelope,
 * injected cells and destroyed cells are visible as what they are - cells.
 */

export const PALETTE = Object.freeze({
  bg: [246, 247, 243],
  grid: [225, 229, 219],
  ring: [211, 217, 205],
  honest: [18, 124, 138],
  injected: [221, 59, 59],
  removed: [224, 145, 43],
  replaced: [176, 92, 214],
  ego: [42, 46, 40],
  spoofer: [221, 59, 59],
  reach: [221, 59, 59],
  box: [110, 120, 104],
  boxHit: [221, 59, 59],
  text: "#5C6359",
  faint: "#9AA093",
});

const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

/** World (metres, sensor frame) -> canvas pixels, ego centred. */
export function makeView(canvas, rangeM) {
  const W = canvas.width;
  const H = canvas.height;
  const cx = W / 2;
  const cy = H / 2;
  const scale = Math.min(W, H) / 2 / rangeM;
  return {
    W,
    H,
    cx,
    cy,
    scale,
    rangeM,
    // x forward (up on screen), y left (left on screen): standard BEV
    px: (x, y) => [cx - y * scale, cy - x * scale],
    inv: (sx, sy) => [(cy - sy) / scale, (cx - sx) / scale],
  };
}

function splat(img, W, H, sx, sy, colour, alpha, size) {
  const x0 = Math.max(0, Math.round(sx) - size);
  const x1 = Math.min(W - 1, Math.round(sx) + size);
  const y0 = Math.max(0, Math.round(sy) - size);
  const y1 = Math.min(H - 1, Math.round(sy) + size);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const o = (y * W + x) * 4;
      img[o] = img[o] + (colour[0] - img[o]) * alpha;
      img[o + 1] = img[o + 1] + (colour[1] - img[o + 1]) * alpha;
      img[o + 2] = img[o + 2] + (colour[2] - img[o + 2]) * alpha;
      img[o + 3] = 255;
    }
  }
}

/**
 * Bird's-eye view of a frame, optionally perturbed.
 * `keepMask` 0 marks a destroyed return; pass showRemoved to draw those in
 * place so the user can see what the attack deleted.
 */
export function renderBEV(canvas, {
  frame,
  keepMask = null,
  spoof = null,
  replacement = null,
  boxes = [],
  reach = null,
  spooferPos = null,
  rangeM = 80,
  showRemoved = true,
  pointSize = 0,
}) {
  const ctx = canvas.getContext("2d");
  const v = makeView(canvas, rangeM);
  const { W, H } = v;

  const imageData = ctx.createImageData(W, H);
  const img = imageData.data;
  for (let i = 0; i < img.length; i += 4) {
    img[i] = PALETTE.bg[0];
    img[i + 1] = PALETTE.bg[1];
    img[i + 2] = PALETTE.bg[2];
    img[i + 3] = 255;
  }

  // real returns
  for (let i = 0; i < frame.count; i++) {
    const destroyed = keepMask !== null && keepMask[i] === 0;
    if (destroyed && !showRemoved) continue;
    const [sx, sy] = v.px(frame.x[i], frame.y[i]);
    if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
    splat(
      img, W, H, sx, sy,
      destroyed ? PALETTE.removed : PALETTE.honest,
      destroyed ? 0.95 : 0.55,
      destroyed ? 1 : pointSize
    );
  }
  // range-replaced returns
  if (replacement) {
    for (let i = 0; i < replacement.x.length; i++) {
      const [sx, sy] = v.px(replacement.x[i], replacement.y[i]);
      if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
      splat(img, W, H, sx, sy, PALETTE.replaced, 1, 1);
    }
  }
  // injected phantom
  if (spoof) {
    for (let i = 0; i < spoof.x.length; i++) {
      const [sx, sy] = v.px(spoof.x[i], spoof.y[i]);
      if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
      splat(img, W, H, sx, sy, PALETTE.injected, 1, 1);
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // --- vector overlays ---
  ctx.save();
  ctx.strokeStyle = rgb(PALETTE.ring);
  ctx.lineWidth = 1;
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = PALETTE.faint;
  for (let k = 1; k <= 4; k++) {
    const r = (rangeM * k) / 4;
    ctx.beginPath();
    ctx.arc(v.cx, v.cy, r * v.scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillText(r.toFixed(0) + "m", v.cx + 3, v.cy - r * v.scale + 11);
  }

  // reachable wedge
  if (reach && reach.feasible && spooferPos) {
    const g = reach.geometry;
    const half = reach.envelope.halfWidthAz;
    ctx.fillStyle = rgb(PALETTE.reach, 0.1);
    ctx.strokeStyle = rgb(PALETTE.reach, 0.4);
    ctx.beginPath();
    ctx.moveTo(v.cx, v.cy);
    const R = Math.min(v.W, v.H);
    for (let a = -half; a <= half; a += Math.max(half / 24, 1e-4)) {
      const ang = g.bearing + a;
      const [sx, sy] = v.px(Math.cos(ang) * rangeM * 1.5, Math.sin(ang) * rangeM * 1.5);
      ctx.lineTo(sx, sy);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // ground-truth boxes
  for (const b of boxes) {
    const c = Math.cos(b.yaw);
    const s = Math.sin(b.yaw);
    const hx = b.size[0] / 2;
    const hy = b.size[1] / 2;
    ctx.strokeStyle = rgb(PALETTE.box, 0.75);
    ctx.lineWidth = 1;
    ctx.beginPath();
    [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].forEach(([lx, ly], i) => {
      const wx = b.center[0] + lx * c - ly * s;
      const wy = b.center[1] + lx * s + ly * c;
      const [sx, sy] = v.px(wx, wy);
      if (i) ctx.lineTo(sx, sy);
      else ctx.moveTo(sx, sy);
    });
    ctx.closePath();
    ctx.stroke();
  }

  // ego
  ctx.fillStyle = rgb(PALETTE.ego);
  ctx.beginPath();
  ctx.moveTo(v.cx, v.cy - 7);
  ctx.lineTo(v.cx - 5, v.cy + 5);
  ctx.lineTo(v.cx + 5, v.cy + 5);
  ctx.closePath();
  ctx.fill();

  // spoofer
  if (spooferPos) {
    const [sx, sy] = v.px(spooferPos[0], spooferPos[1]);
    const blocked = reach && !reach.lineOfSight.clear;
    ctx.fillStyle = rgb(PALETTE.spoofer, blocked ? 0.35 : 1);
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = rgb(PALETTE.spoofer, blocked ? 0.35 : 0.9);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(sx, sy, 9, 0, Math.PI * 2);
    ctx.stroke();
    if (blocked) {
      ctx.strokeStyle = rgb(PALETTE.ego, 0.7);
      ctx.beginPath();
      ctx.moveTo(sx - 7, sy - 7);
      ctx.lineTo(sx + 7, sy + 7);
      ctx.stroke();
    }
  }
  ctx.restore();
}

/**
 * The scan lattice itself: azimuth columns across, beams down, windowed around
 * the spoofer bearing. This is where the per-cell physics becomes visible.
 */
export function renderRangeImage(canvas, {
  frame,
  sensor,
  keepMask = null,
  reach = null,
  spoofCells = null,
  windowDeg = 25,
}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = rgb(PALETTE.bg);
  ctx.fillRect(0, 0, W, H);

  const centreAz = reach && reach.feasible ? reach.geometry.bearing : 0;
  const halfWin = (windowDeg * Math.PI) / 180;
  const colFrom = Math.round((centreAz - halfWin) / sensor.azRes);
  const colTo = Math.round((centreAz + halfWin) / sensor.azRes);
  const nCols = colTo - colFrom;
  const cw = W / nCols;
  const ch = H / sensor.nBeams;
  const wrap = (c) => ((c % sensor.azCellCount) + sensor.azCellCount) % sensor.azCellCount;

  // occupancy: any real return in the cell
  for (let c = colFrom; c < colTo; c++) {
    const col = wrap(c);
    for (let b = 0; b < sensor.nBeams; b++) {
      const key = col * sensor.nBeams + b;
      const { start, end } = frame.cellRange(key);
      if (end === start) continue;
      let alive = false;
      for (let i = start; i < end; i++) {
        if (keepMask === null || keepMask[frame.order[i]]) alive = true;
      }
      const x = (c - colFrom) * cw;
      const y = b * ch;
      ctx.fillStyle = alive ? rgb(PALETTE.honest, 0.55) : rgb(PALETTE.removed, 0.95);
      ctx.fillRect(x, y, Math.max(cw, 1), Math.max(ch - 0.5, 1));
    }
  }

  // reachable envelope outline
  if (reach && reach.feasible) {
    ctx.fillStyle = rgb(PALETTE.reach, 0.10);
    for (let i = 0; i < reach.cells.length; i++) {
      const col = Math.floor(reach.cells[i] / sensor.nBeams);
      const b = reach.cells[i] - col * sensor.nBeams;
      let c = col;
      if (c - colFrom < -sensor.azCellCount / 2) c += sensor.azCellCount;
      if (c - colFrom > sensor.azCellCount / 2) c -= sensor.azCellCount;
      if (c < colFrom || c >= colTo) continue;
      ctx.fillRect((c - colFrom) * cw, b * ch, Math.max(cw, 1), Math.max(ch - 0.5, 1));
    }
  }

  // injected cells
  if (spoofCells) {
    ctx.fillStyle = rgb(PALETTE.injected, 0.95);
    for (let i = 0; i < spoofCells.length; i++) {
      const col = Math.floor(spoofCells[i] / sensor.nBeams);
      const b = spoofCells[i] - col * sensor.nBeams;
      let c = col;
      if (c - colFrom < -sensor.azCellCount / 2) c += sensor.azCellCount;
      if (c - colFrom > sensor.azCellCount / 2) c -= sensor.azCellCount;
      if (c < colFrom || c >= colTo) continue;
      ctx.fillRect((c - colFrom) * cw, b * ch, Math.max(cw, 1), Math.max(ch - 0.5, 1));
    }
  }

  // axes
  ctx.strokeStyle = rgb(PALETTE.grid);
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  ctx.fillStyle = PALETTE.faint;
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`azimuth ${(-windowDeg).toFixed(0)}°`, 4, H - 4);
  ctx.textAlign = "right";
  ctx.fillText(`+${windowDeg.toFixed(0)}° (bearing-relative)`, W - 4, H - 4);
  ctx.textAlign = "left";
  ctx.fillText(`beam 0`, 4, 11);
  ctx.fillText(`beam ${sensor.nBeams - 1}`, 4, H - 16);
}
