/**
 * Rendering.
 *
 * Three deliberate choices:
 *
 *  1. The palette is a hoisted constant. The MVP called getComputedStyle() *per
 *     point* inside the draw loop, forcing thousands of style recalcs a frame.
 *  2. Points are splatted into an ImageData buffer rather than drawn as arc()
 *     paths, and the background is cleared through a Uint32 view (one write per
 *     pixel instead of four). At 120k points and 60 fps that is the difference
 *     between an interactive view and a slideshow.
 *  3. The sweep reveals points in true azimuth order by walking a prefix of
 *     `frame.order`, which is already azimuth-sorted. The animation therefore
 *     shows the order the sensor really acquired the cloud, not a decoration.
 *  4. The display PERSISTS: it does not blank at the end of a revolution. Each
 *     azimuth column keeps whatever the beam last saw there, so the screen is the
 *     current frame ahead of the beam and the previous frame behind it. On a
 *     static frame the two layers reassemble one cloud; in a sequence they are two
 *     keyframes, which is the mosaic a spinning sensor really produces.
 *
 * ATLAS's removal sector is drawn as what it is - a hard rectangle in azimuth,
 * uniform inside - and is visually distinct from the v1 physics envelope, which
 * is a power-derived Gaussian. Conflating the two on screen would misrepresent
 * which model produced the attack.
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
  atlasSector: [147, 51, 234],
  box: [110, 120, 104],
  target: [221, 59, 59],
  phantom: [221, 59, 59],
  beam: [118, 185, 0],
  text: "#5C6359",
  faint: "#9AA093",
});

const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const packBg = (c) => (255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];

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
  const x0 = Math.max(0, (sx | 0) - size);
  const x1 = Math.min(W - 1, (sx | 0) + size);
  const y0 = Math.max(0, (sy | 0) - size);
  const y1 = Math.min(H - 1, (sy | 0) + size);
  for (let y = y0; y <= y1; y++) {
    let o = (y * W + x0) * 4;
    for (let x = x0; x <= x1; x++, o += 4) {
      img[o] += (colour[0] - img[o]) * alpha;
      img[o + 1] += (colour[1] - img[o + 1]) * alpha;
      img[o + 2] += (colour[2] - img[o + 2]) * alpha;
      img[o + 3] = 255;
    }
  }
}

function strokeBox(ctx, v, box, colour, width = 1, dash = null) {
  const c = Math.cos(box.yaw ?? 0);
  const s = Math.sin(box.yaw ?? 0);
  const hx = box.size[0] / 2;
  const hy = box.size[1] / 2;
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].forEach(([lx, ly], i) => {
    const wx = box.center[0] + lx * c - ly * s;
    const wy = box.center[1] + lx * s + ly * c;
    const [px, py] = v.px(wx, wy);
    if (i) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
  });
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

/**
 * Bird's-eye view of a frame, optionally perturbed and optionally mid-sweep.
 *
 * sweep: {cutoff, azIdx, az} reveals only the points the beam has passed.
 *        Pass null to show the whole revolution at once.
 */
export function renderBEV(canvas, {
  frame,
  keepMask = null,
  spoof = null,
  replacement = null,
  boxes = [],
  reach = null,
  atlasSector = null,
  laser = null,
  spooferPos = null,
  targetBox = null,
  phantomBox = null,
  rangeM = 80,
  showRemoved = true,
  pointSize = 0,
  sweep = null,
  sectorActive = false,
  prevLayer = null,
}) {
  const ctx = canvas.getContext("2d");
  const v = makeView(canvas, rangeM);
  const { W, H } = v;

  const imageData = ctx.createImageData(W, H);
  const img = imageData.data;
  new Uint32Array(imageData.data.buffer).fill(packBg(PALETTE.bg));

  // --- real returns ---
  // Persistent display: the columns behind the beam come from the previous layer,
  // the columns it has already passed this revolution from the current frame.
  const drawReturns = (f, mask, from, to) => {
    for (let k = from; k < to; k++) {
      const i = f.order[k];
      const destroyed = mask !== null && mask[i] === 0;
      if (destroyed && !showRemoved) continue;
      const [sx, sy] = v.px(f.x[i], f.y[i]);
      if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
      splat(
        img, W, H, sx, sy,
        destroyed ? PALETTE.removed : PALETTE.honest,
        destroyed ? 0.95 : 0.55,
        destroyed ? 1 : pointSize
      );
    }
  };

  if (!sweep) {
    drawReturns(frame, keepMask, 0, frame.count);
  } else {
    if (prevLayer && prevLayer.frame) {
      drawReturns(prevLayer.frame, prevLayer.keepMask ?? null, prevLayer.prevStart, prevLayer.frame.count);
    }
    drawReturns(frame, keepMask, 0, Math.min(sweep.cutoff, frame.count));
  }

  // Added points (phantom / range-replaced) follow the same rule: ahead of the
  // beam they belong to the previous layer, behind it to the current frame.
  const splatSet = (set, colour) => {
    if (!set) return;
    for (let i = 0; i < set.x.length; i++) {
      const [sx, sy] = v.px(set.x[i], set.y[i]);
      if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
      splat(img, W, H, sx, sy, colour, 1, 1);
    }
  };
  const azOf = (set, i) => {
    const a = set.az ? set.az[i] : Math.atan2(set.y[i], set.x[i]);
    return a < 0 ? a + 2 * Math.PI : a;
  };
  const splitSet = (set, colour) => {
    if (!set) return;
    for (let i = 0; i < set.x.length; i++) {
      if (azOf(set, i) > sweep.az) continue; // not yet re-scanned this revolution
      const [sx, sy] = v.px(set.x[i], set.y[i]);
      if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
      splat(img, W, H, sx, sy, colour, 1, 1);
    }
  };
  const splitSetAhead = (set, colour) => {
    if (!set) return;
    for (let i = 0; i < set.x.length; i++) {
      if (azOf(set, i) <= sweep.az) continue; // already refreshed by the current frame
      const [sx, sy] = v.px(set.x[i], set.y[i]);
      if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
      splat(img, W, H, sx, sy, colour, 1, 1);
    }
  };

  if (!sweep) {
    splatSet(replacement, PALETTE.replaced);
    splatSet(spoof, PALETTE.injected);
  } else {
    if (prevLayer) {
      splitSetAhead(prevLayer.replacement, PALETTE.replaced);
      splitSetAhead(prevLayer.spoof, PALETTE.injected);
    }
    splitSet(replacement, PALETTE.replaced);
    splitSet(spoof, PALETTE.injected);
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

  // ATLAS's removal sector: a hard rectangle in azimuth, uniform inside
  if (atlasSector) {
    const half = (atlasSector.azWidthDeg * Math.PI) / 180 / 2;
    const R = rangeM * 1.6;
    ctx.fillStyle = rgb(PALETTE.atlasSector, sectorActive ? 0.2 : 0.1);
    ctx.strokeStyle = rgb(PALETTE.atlasSector, sectorActive ? 0.85 : 0.45);
    ctx.lineWidth = sectorActive ? 1.6 : 1;
    ctx.beginPath();
    ctx.moveTo(v.cx, v.cy);
    for (let a = -half; a <= half; a += Math.max(half / 32, 1e-4)) {
      const ang = atlasSector.azCenter + a;
      const [sx, sy] = v.px(Math.cos(ang) * R, Math.sin(ang) * R);
      ctx.lineTo(sx, sy);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // v1 physics envelope (power-derived, Gaussian) - only when the overlay is on
  if (reach && reach.feasible && spooferPos) {
    const half = reach.envelope.halfWidthAz;
    ctx.fillStyle = rgb(PALETTE.reach, 0.09);
    ctx.strokeStyle = rgb(PALETTE.reach, 0.35);
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(v.cx, v.cy);
    for (let a = -half; a <= half; a += Math.max(half / 24, 1e-4)) {
      const ang = reach.geometry.bearing + a;
      const [sx, sy] = v.px(Math.cos(ang) * rangeM * 1.5, Math.sin(ang) * rangeM * 1.5);
      ctx.lineTo(sx, sy);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ground-truth boxes
  for (const b of boxes) strokeBox(ctx, v, b, rgb(PALETTE.box, 0.7), 1);
  if (targetBox) strokeBox(ctx, v, targetBox, rgb(PALETTE.target, 0.95), 2.2);
  if (phantomBox) strokeBox(ctx, v, phantomBox, rgb(PALETTE.phantom, 0.9), 1.8, [5, 3]);

  // The attacker's hardware, firing at the victim's receiver.
  //
  // ATLAS does not model an attacker position at all - injection is a trace
  // placed in the cloud, removal is a sector aimed at a target. To draw the
  // hardware at all we have to choose a position, so it sits on the sector's own
  // bearing: that is the only placement for which the laser path and the cells it
  // affects are geometrically consistent. The attack applied to the cloud is
  // unchanged by it.
  //
  // The beam is aimed at the ego's aperture, not sprayed into the scene, and it
  // only matters while the victim's beam is pointed back at it - which is what
  // "active" encodes.
  if (laser) {
    const [lx, ly] = v.px(laser.pos[0], laser.pos[1]);
    const active = laser.active;

    ctx.save();
    // the attack path
    ctx.strokeStyle = rgb(PALETTE.injected, active ? 0.95 : 0.22);
    ctx.lineWidth = active ? 2.4 : 1;
    if (!active) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(v.cx, v.cy);
    ctx.stroke();
    ctx.setLineDash([]);

    if (active) {
      // pulses in flight toward the receiver. The real train is at tens of kHz,
      // far faster than a display frame, so these are indicative of firing, not
      // individual pulses to be counted.
      const nPulses = 5;
      const phase = laser.phase ?? 0;
      for (let i = 0; i < nPulses; i++) {
        const t = ((i / nPulses + phase) % 1);
        const px = lx + (v.cx - lx) * t;
        const py = ly + (v.cy - ly) * t;
        ctx.fillStyle = rgb(PALETTE.injected, 0.9 * (1 - Math.abs(0.5 - t) * 0.8));
        ctx.beginPath();
        ctx.arc(px, py, 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
      // the receiver being illuminated
      const glow = ctx.createRadialGradient(v.cx, v.cy, 0, v.cx, v.cy, 22);
      glow.addColorStop(0, rgb(PALETTE.injected, 0.5));
      glow.addColorStop(1, rgb(PALETTE.injected, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(v.cx, v.cy, 22, 0, Math.PI * 2);
      ctx.fill();
    }

    // the hardware itself
    ctx.fillStyle = rgb(PALETTE.injected, active ? 1 : 0.55);
    ctx.beginPath();
    ctx.arc(lx, ly, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = rgb(PALETTE.injected, active ? 1 : 0.5);
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(lx, ly, active ? 12 : 9.5, 0, Math.PI * 2);
    ctx.stroke();
    if (active) {
      ctx.beginPath();
      ctx.arc(lx, ly, 17, 0, Math.PI * 2);
      ctx.strokeStyle = rgb(PALETTE.injected, 0.35);
      ctx.stroke();
    }
    ctx.restore();
  }

  // the rotating beam
  if (sweep) {
    const R = Math.min(W, H) / 2;
    ctx.save();
    ctx.translate(v.cx, v.cy);
    for (let i = 0; i < 14; i++) {
      const a = sweep.az - i * 0.045;
      ctx.strokeStyle = rgb(PALETTE.beam, 0.18 * (1 - i / 14));
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      // screen y is inverted and x is -y, so rotate accordingly
      ctx.lineTo(-Math.sin(a) * R, -Math.cos(a) * R);
      ctx.stroke();
    }
    ctx.strokeStyle = rgb(PALETTE.beam, 0.9);
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-Math.sin(sweep.az) * R, -Math.cos(sweep.az) * R);
    ctx.stroke();
    ctx.restore();
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
 * the attack's aim point. This is where the per-cell view becomes legible.
 */
export function renderRangeImage(canvas, {
  frame,
  sensor,
  keepMask = null,
  centerAz = 0,
  reach = null,
  spoofCells = null,
  sectorMask = null,
  windowDeg = 25,
  sweep = null,
}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = rgb(PALETTE.bg);
  ctx.fillRect(0, 0, W, H);

  const halfWin = (windowDeg * Math.PI) / 180;
  const colFrom = Math.round((centerAz - halfWin) / sensor.azRes);
  const colTo = Math.round((centerAz + halfWin) / sensor.azRes);
  const nCols = Math.max(1, colTo - colFrom);
  const cw = W / nCols;
  const ch = H / sensor.nBeams;
  const wrap = (c) => ((c % sensor.azCellCount) + sensor.azCellCount) % sensor.azCellCount;

  for (let c = colFrom; c < colTo; c++) {
    const col = wrap(c);
    for (let b = 0; b < sensor.nBeams; b++) {
      const key = col * sensor.nBeams + b;
      const { start, end } = frame.cellRange(key);
      if (end === start) continue;
      let alive = false;
      let inSector = false;
      for (let i = start; i < end; i++) {
        const p = frame.order[i];
        if (keepMask === null || keepMask[p]) alive = true;
        if (sectorMask && sectorMask[p]) inSector = true;
      }
      const x = (c - colFrom) * cw;
      const y = b * ch;
      ctx.fillStyle = alive
        ? rgb(PALETTE.honest, inSector ? 0.8 : 0.5)
        : rgb(PALETTE.removed, 0.95);
      ctx.fillRect(x, y, Math.max(cw, 1), Math.max(ch - 0.5, 1));
    }
  }

  if (reach && reach.feasible) {
    ctx.fillStyle = rgb(PALETTE.reach, 0.1);
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

  // where the beam currently is, in lattice coordinates
  if (sweep) {
    let c = Math.floor(sweep.az / sensor.azRes);
    if (c - colFrom < -sensor.azCellCount / 2) c += sensor.azCellCount;
    if (c - colFrom > sensor.azCellCount / 2) c -= sensor.azCellCount;
    if (c >= colFrom && c < colTo) {
      ctx.strokeStyle = rgb(PALETTE.beam, 0.9);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo((c - colFrom) * cw, 0);
      ctx.lineTo((c - colFrom) * cw, H);
      ctx.stroke();
    }
  }

  ctx.strokeStyle = rgb(PALETTE.grid);
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  ctx.fillStyle = PALETTE.faint;
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`${(-windowDeg).toFixed(0)}°`, 4, H - 4);
  ctx.textAlign = "right";
  ctx.fillText(`+${windowDeg.toFixed(0)}° from aim point`, W - 4, H - 4);
  ctx.textAlign = "left";
  ctx.fillText("beam 0", 4, 11);
  ctx.fillText(`beam ${sensor.nBeams - 1}`, 4, H - 16);
}
