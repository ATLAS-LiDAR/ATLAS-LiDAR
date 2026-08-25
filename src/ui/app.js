/**
 * ATLAS visualizer orchestration (v2).
 *
 * Defaults to ATLAS's own algorithms with nothing added:
 *   inject  - placeTraceRelative over a real trace library, then spoof_frame
 *             occlusion, ground anchoring and range-conditioned intensity
 *   remove  - ahfrRemoval, the A-HFR roadside spoofer aimed at a chosen vehicle
 *
 * The v1 physics model (power envelope, line of sight, time of flight, sync) is
 * an OPT-IN overlay, because ATLAS applies none of it. Keeping it off by default
 * is what makes this a visualizer of ATLAS rather than of something adjacent.
 *
 * Two caches keep it interactive on 120k-point frames: the loaded cloud per
 * scene, and the lattice index per scene+sensor. The attack is recomputed only
 * when a parameter changes; the animation just re-renders with a new sweep
 * position, which is an O(1) prefix lookup into the already azimuth-sorted index.
 */
import { configureSensor, listSensors, SENSORS } from "../core/sensor.js";
import { configureSpoofer, listSpoofers, SPOOFERS, reachableCells } from "../core/spoofer.js";
import { injectPhantom } from "../core/inject.js";
import { ahfrRemoval, selectInitialTarget, AHFR_DEFAULTS, sectorMask } from "../core/atlas-removal.js";
import { placeTraceRelative } from "../core/atlas-inject.js";
import { loadTraceLibrary, loadTraceManifest, allTraces } from "../io/traces.js";
import { loadAtlasFrame, loadManifest, parseAtlasFrame, parseRawLidar, toPointFrame } from "../io/atlasframe.js";
import { composeCloud } from "../core/compose.js";
import { buildAttackBundle, downloadBundle, bundleFilename } from "../io/export.js";
import { pickBoxAt, normalizeBox } from "../core/iou.js";
import { sweptPointCount, advanceSweep, sweepAzToCell, beamInSector } from "../core/sweep.js";
import { renderBEV, renderRangeImage, makeView } from "./render.js";

const DEG = Math.PI / 180;
const $ = (id) => document.getElementById(id);

/** GT label -> the integer class column ATLAS's target selection expects. */
function labelToClass(label) {
  if (/car|van|truck|bus|trailer|tram|vehicle/i.test(label)) return 1;
  if (/pedestrian|person/i.test(label)) return 2;
  if (/cyclist|bicycle|motorcycle/i.test(label)) return 3;
  return 0;
}

const S = {
  manifest: [],
  traceManifest: [],
  traceLibs: new Map(),
  sceneSlug: null,
  loaded: null,
  frame: null,
  frameKey: null,
  sensorId: "hdl64e",
  hz: 10,
  attack: "inject",
  difficulty: "easy",
  seed: 7,
  intensityMode: "range-conditioned",
  groundAnchor: true,
  targetIndex: -1,
  azWidthDeg: AHFR_DEFAULTS.azWidthDeg,
  pRemove: AHFR_DEFAULTS.pRemove,
  forceActive: false,
  overlay: false,
  spooferId: "cao2023",
  spooferPos: [25, 6, 0],
  spinning: true,
  sweepAz: 0,
  speed: 0.125,
  viewRange: 70,
  showRemoved: true,
  showBoxes: true,
  last: null,
};

function setStatus(msg, cls = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

const sensor = () => configureSensor(S.sensorId, { hz: S.hz });
const spoofer = () => configureSpoofer(S.spooferId);

/** GT boxes as ATLAS 8-vectors [x,y,z,dx,dy,dz,yaw,class]. */
function atlasBoxes() {
  return (S.loaded?.header.boxes || []).map((b) => [
    b.center[0], b.center[1], b.center[2],
    b.size[0], b.size[1], b.size[2], b.yaw,
    labelToClass(b.label),
  ]);
}

function targetBox() {
  const boxes = S.loaded?.header.boxes || [];
  if (S.targetIndex < 0 || S.targetIndex >= boxes.length) return null;
  const b = boxes[S.targetIndex];
  return [b.center[0], b.center[1], b.center[2], b.size[0], b.size[1], b.size[2], b.yaw];
}

/* ------------------------------ data loading ------------------------------ */

async function traceLib() {
  const dataset = S.loaded?.header.dataset === "kitti" ? "kitti" : "nuscenes";
  const key = dataset + "-" + S.difficulty;
  if (!S.traceLibs.has(key)) {
    const lib = await loadTraceLibrary(`assets/traces/${key}.atlastrace`);
    S.traceLibs.set(key, { lib, traces: allTraces(lib) });
  }
  return S.traceLibs.get(key);
}

function ensureFrame() {
  const sn = sensor();
  const key = `${S.sceneSlug}|${sn.id}|${sn.hz}`;
  if (S.frame && S.frameKey === key) return S.frame;
  S.frame = toPointFrame(S.loaded, sn);
  S.frameKey = key;
  return S.frame;
}

async function selectScene(slug) {
  const entry = S.manifest.find((m) => m.slug === slug);
  if (!entry) return;
  setStatus("Loading " + entry.label + "…");
  S.loaded = await loadAtlasFrame("assets/frames/" + entry.file);
  S.sceneSlug = slug;
  if (SENSORS[S.loaded.header.sensor]) {
    S.sensorId = S.loaded.header.sensor;
    $("sensor").value = S.sensorId;
  }
  S.frame = null;
  ensureFrame();
  autoTarget(false);
  await traceLib();
  labels();
  await recompute();
}

/** ATLAS's own target choice. */
function autoTarget(announce = true) {
  const { index } = selectInitialTarget(atlasBoxes(), {
    desiredRange: AHFR_DEFAULTS.desiredRange,
    tol: AHFR_DEFAULTS.tol,
    targetClass: 1,
    minRange: AHFR_DEFAULTS.minRange,
  });
  S.targetIndex = index == null ? -1 : index;
  updateTargetInfo();
  if (announce) {
    setStatus(
      S.targetIndex < 0
        ? "ATLAS found no eligible target: it needs an in-front vehicle beyond 20 m."
        : `ATLAS selected target #${S.targetIndex}.`,
      S.targetIndex < 0 ? "warn" : "ok"
    );
  }
}

function updateTargetInfo() {
  const boxes = S.loaded?.header.boxes || [];
  const el = $("targetInfo");
  if (S.targetIndex < 0 || !boxes[S.targetIndex]) {
    el.innerHTML = "No target selected — click a vehicle";
    return;
  }
  const b = boxes[S.targetIndex];
  const r = Math.hypot(b.center[0], b.center[1], b.center[2]);
  el.innerHTML = `<b>#${S.targetIndex} ${b.label}</b> at ${r.toFixed(1)} m`;
}

/* ------------------------------- compute --------------------------------- */

async function recompute() {
  if (!S.loaded) return;
  const frame = ensureFrame();
  const sn = frame.sensor;
  const sp = spoofer();
  const t0 = performance.now();

  const reach = S.overlay
    ? reachableCells({ frame, sensor: sn, spoofer: sp, positionXYZ: S.spooferPos })
    : null;

  let result = null;
  let keepMask = null;
  let spoof = null;
  let phantomBox = null;
  let atlasSector = null;
  let secMask = null;
  let centerAz = 0;

  if (S.attack === "inject") {
    const { traces } = await traceLib();
    const placed = placeTraceRelative(traces, { seed: S.seed });
    phantomBox = { center: placed.box.slice(0, 3), size: placed.box.slice(3, 6), yaw: placed.box[6] };
    result = injectPhantom({
      frame,
      sensor: sn,
      spoofer: sp,
      positionXYZ: S.spooferPos,
      phantom: placed.points,
      reach,
      dataset: S.loaded.header.dataset,
      intensityMode: S.intensityMode,
      constantIntensity: 100,
      seed: S.seed,
      groundAnchor: S.groundAnchor,
      atlasFaithful: !S.overlay,
    });
    keepMask = result.keepMask;
    spoof = result.spoof;
    result.placed = placed;
    centerAz = Math.atan2(placed.box[1], placed.box[0]);
    if (centerAz < 0) centerAz += 2 * Math.PI;
  } else {
    const tb = targetBox();
    if (!tb) {
      setStatus("Pick a vehicle to hide — click one in the left view, or use auto-select.", "warn");
      S.last = { sensor: sn, reach, keepMask: null, spoof: null, result: null, phantomBox: null, atlasSector: null, secMask: null, centerAz: 0 };
      draw();
      return;
    }
    result = ahfrRemoval({
      frame,
      targetBox: tb,
      azWidthDeg: S.azWidthDeg,
      pRemove: S.pRemove,
      forceActive: S.forceActive,
      seed: S.seed,
    });
    keepMask = result.keepMask;
    atlasSector = { azCenter: result.meta.azCenterRad, azWidthDeg: S.azWidthDeg };
    secMask = result.meta.sectorMask || sectorMask(frame, result.meta.azCenterRad, result.meta.elCenterRad, S.azWidthDeg, result.meta.elWidthDeg);
    centerAz = result.meta.azCenterRad < 0 ? result.meta.azCenterRad + 2 * Math.PI : result.meta.azCenterRad;

    // In overlay mode, mark returns the physics says are unreachable
    if (S.overlay && reach) {
      const reachable = new Set(Array.from(reach.cells));
      for (let i = 0; i < frame.count; i++) {
        if (!keepMask[i] && !reachable.has(frame.cell[i])) keepMask[i] = 1;
      }
    }
  }

  S.last = { sensor: sn, spoofer: sp, reach, keepMask, spoof, result, phantomBox, atlasSector, secMask, centerAz, ms: performance.now() - t0 };
  draw();
  report();
}

/* -------------------------------- render --------------------------------- */

function draw() {
  if (!S.last || !S.frame) return;
  const { sensor: sn, reach, keepMask, spoof, phantomBox, atlasSector, secMask, centerAz } = S.last;
  const frame = S.frame;

  const sweep = S.spinning
    ? {
        az: S.sweepAz,
        azIdx: sweepAzToCell(S.sweepAz, sn),
        cutoff: sweptPointCount(frame, sweepAzToCell(S.sweepAz, sn)),
      }
    : null;

  const sectorActive = atlasSector
    ? beamInSector(S.sweepAz, atlasSector.azCenter, atlasSector.azWidthDeg * DEG)
    : false;

  const boxes = S.showBoxes ? S.loaded.header.boxes || [] : [];
  const tb = targetBox();
  const targetView = tb && S.attack === "remove"
    ? { center: tb.slice(0, 3), size: tb.slice(3, 6), yaw: tb[6] }
    : null;

  const common = { frame, boxes, reach, spooferPos: S.overlay ? S.spooferPos : null, rangeM: S.viewRange, sweep };
  renderBEV($("cvClean"), { ...common, keepMask: null, spoof: null, showRemoved: false, targetBox: targetView });
  renderBEV($("cvPert"), {
    ...common,
    keepMask,
    spoof,
    atlasSector,
    sectorActive,
    targetBox: targetView,
    phantomBox,
    showRemoved: S.showRemoved,
  });
  renderRangeImage($("cvLattice"), {
    frame,
    sensor: sn,
    keepMask,
    centerAz,
    reach,
    spoofCells: spoof ? spoof.cell : null,
    sectorMask: secMask,
    sweep,
  });
}

function stat(k, v, cls = "") {
  return `<div class="stat ${cls}"><div class="k2">${k}</div><div class="v">${v}</div></div>`;
}

function report() {
  if (!S.last || !S.last.result) {
    $("readout").innerHTML = "";
    return;
  }
  const { result, reach, ms } = S.last;
  const rows = [stat("Frame points", S.frame.count.toLocaleString())];

  if (S.attack === "inject") {
    const st = result.stats;
    const p = result.placed;
    rows.push(
      stat("Phantom points", st.injected.toLocaleString()),
      stat("Trace", `#${p.anno.traceIdx}<small> / ${p.anno.nTracePoints}pt</small>`),
      stat("Placed at", `${p.box[0].toFixed(1)}<small>,${p.box[1].toFixed(1)} m</small>`),
      stat("Yaw", `${(p.anno.yaw / DEG).toFixed(1)}<small>°</small>`),
      stat("Occluded away", st.rejected.occluded.toLocaleString()),
      stat("Real returns killed", st.suppressed.toLocaleString()),
      stat("Returns / cell", st.fill.toFixed(2), st.fill > 1 ? "warn2" : "good"),
      stat("Ground shift", `${st.groundShiftZ.toFixed(2)}<small> m</small>`)
    );
    if (S.overlay) {
      const r = st.rejected;
      rows.push(
        stat("Unreachable", r.unreachable, r.unreachable ? "warn2" : ""),
        stat("Weak response", r.response),
        stat("ToF rejected", r.tof, r.tof ? "warn2" : ""),
        stat("Sync miss", r.sync)
      );
    }
    const share = st.requested ? ((st.injected / st.requested) * 100).toFixed(0) : "0";
    setStatus(
      `Injected ${st.injected}/${st.requested} trace points (${share}%) from trace #${p.anno.traceIdx}` +
        ` at ${p.box[0].toFixed(1)}, ${p.box[1].toFixed(1)} m` +
        ` · ${st.rejected.occluded} occluded, ${st.suppressed} real returns destroyed` +
        (S.overlay ? " · physics overlay ON (not ATLAS)" : "") +
        ` · ${ms.toFixed(0)} ms`,
      st.injected === 0 ? "warn" : "ok"
    );
  } else {
    const m = result.meta;
    rows.push(
      stat("In sector", m.nPointsInSector.toLocaleString()),
      stat("Destroyed", m.nPointsRemoved.toLocaleString()),
      stat("Removal rate", m.nPointsInSector ? (m.nPointsRemoved / m.nPointsInSector).toFixed(2) : "—"),
      stat("Sector", `${m.azWidthDeg}<small>°×${m.elWidthDeg}°</small>`),
      stat("Aim", `${(m.azCenterRad / DEG).toFixed(1)}<small>°</small>`),
      stat("Target range", `${m.targetRangeM.toFixed(1)}<small> m</small>`),
      stat("Scan gate p", m.scanGateProb.toFixed(2)),
      stat("Active frame", m.activeThisFrame ? "yes" : "no", m.activeThisFrame ? "good" : "warn2")
    );
    const why = {
      scan_gate: `the frame-level scan gate did not fire (p=${m.scanGateProb.toFixed(2)} at ${m.targetRangeM.toFixed(0)} m) — ATLAS models the attacker missing launches`,
      target_too_close: `the target is inside ATLAS's ${AHFR_DEFAULTS.minTargetRangeM} m minimum range`,
      empty_sector: "no returns fall inside the sector",
    }[m.inactiveReason];
    setStatus(
      m.inactiveReason
        ? `No removal this frame: ${why}.`
        : `Destroyed ${m.nPointsRemoved} of ${m.nPointsInSector} returns in the ${m.azWidthDeg}°×${m.elWidthDeg}° sector` +
            ` (rate ${(m.nPointsRemoved / m.nPointsInSector).toFixed(2)}, p_remove ${m.pRemove})` +
            ` · aim ${(m.azCenterRad / DEG).toFixed(1)}°` +
            (S.overlay ? " · physics overlay ON (not ATLAS)" : "") +
            ` · ${ms.toFixed(0)} ms`,
      m.inactiveReason ? "warn" : "ok"
    );
  }
  $("readout").innerHTML = rows.join("");
}

/* ------------------------------- animation ------------------------------- */

let lastT = 0;
function loop(t) {
  const dt = lastT ? Math.min(0.05, (t - lastT) / 1000) : 0;
  lastT = t;
  if (S.spinning && S.frame && S.last) {
    S.sweepAz = advanceSweep(S.sweepAz, dt, S.last.sensor, S.speed);
    draw();
  }
  requestAnimationFrame(loop);
}

/* -------------------------------- wiring --------------------------------- */

function labels() {
  const sn = sensor();
  $("hzVal").textContent = S.hz + " Hz";
  $("sensorSpec").textContent = `${sn.nBeams}b · ${(sn.azRes / DEG).toFixed(3)}°`;
  $("azwVal").textContent = S.azWidthDeg + "°";
  $("prVal").textContent = S.pRemove.toFixed(2);
  $("speedVal").textContent = S.speed.toFixed(2) + "×";
  $("rangeVal").textContent = S.viewRange + " m";
  const e = S.manifest.find((m) => m.slug === S.sceneSlug);
  $("sceneSpec").textContent = e ? `${(e.nPoints / 1000).toFixed(0)}k pts` : "";
  const tm = S.traceManifest.find(
    (m) => m.difficulty === S.difficulty && m.dataset === (S.loaded?.header.dataset === "kitti" ? "kitti" : "nuscenes")
  );
  $("diffSpec").textContent = tm ? `med ${tm.medianPoints} pts` : "";
}

function seg(ids, onPick) {
  ids.forEach((id, i) => {
    $(id).onclick = () => {
      ids.forEach((o, j) => $(o).setAttribute("aria-pressed", String(i === j)));
      onPick(i);
    };
  });
}

export async function boot() {
  for (const id of listSensors()) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = SENSORS[id].name;
    $("sensor").appendChild(o);
  }
  for (const id of listSpoofers()) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = SPOOFERS[id].name;
    $("spooferPreset").appendChild(o);
  }

  S.manifest = await loadManifest("assets/frames/manifest.json");
  S.traceManifest = await loadTraceManifest("assets/traces/manifest.json");
  for (const m of S.manifest) {
    const o = document.createElement("option");
    o.value = m.slug;
    o.textContent = m.label;
    $("scene").appendChild(o);
  }

  $("scene").onchange = (e) => selectScene(e.target.value);
  $("sensor").onchange = (e) => {
    S.sensorId = e.target.value;
    S.frame = null;
    labels();
    recompute();
  };
  $("spooferPreset").onchange = (e) => {
    S.spooferId = e.target.value;
    recompute();
  };

  seg(["atkInject", "atkRemove"], (i) => {
    S.attack = i === 0 ? "inject" : "remove";
    $("injectOpts").style.display = i === 0 ? "" : "none";
    $("removeOpts").style.display = i === 0 ? "none" : "";
    recompute();
  });
  seg(["dEasy", "dMedium", "dHard"], (i) => {
    S.difficulty = ["easy", "medium", "hard"][i];
    labels();
    recompute();
  });
  seg(["playBtn", "fullBtn"], (i) => {
    S.spinning = i === 0;
    draw();
  });

  const bind = (id, apply, tf = (v) => v) => {
    $(id).oninput = (e) => {
      apply(tf(parseFloat(e.target.value)));
      labels();
      recompute();
    };
  };
  bind("hz", (v) => { S.hz = v; S.frame = null; });
  bind("azWidth", (v) => { S.azWidthDeg = v; });
  bind("pRemove", (v) => { S.pRemove = v; });
  bind("viewRange", (v) => { S.viewRange = v; });
  $("speed").oninput = (e) => {
    S.speed = Math.pow(10, parseFloat(e.target.value));
    labels();
  };

  $("intensityMode").onchange = (e) => { S.intensityMode = e.target.value; recompute(); };
  $("groundAnchor").onchange = (e) => { S.groundAnchor = e.target.checked; recompute(); };
  $("forceActive").onchange = (e) => { S.forceActive = e.target.checked; recompute(); };
  $("showRemoved").onchange = (e) => { S.showRemoved = e.target.checked; draw(); };
  $("showBoxes").onchange = (e) => { S.showBoxes = e.target.checked; draw(); };
  $("overlay").onchange = (e) => {
    S.overlay = e.target.checked;
    $("spooferRow").style.display = S.overlay ? "" : "none";
    recompute();
  };
  $("reroll").onclick = () => { S.seed = (S.seed * 1103515245 + 12345) & 0x7fffffff; recompute(); };
  $("autoTarget").onclick = () => { autoTarget(true); recompute(); };

  // click: hit a vehicle -> aim at it; otherwise move the spoofer
  const cv = $("cvClean");
  cv.onclick = (ev) => {
    const r = cv.getBoundingClientRect();
    const sx = ((ev.clientX - r.left) * cv.width) / r.width;
    const sy = ((ev.clientY - r.top) * cv.height) / r.height;
    const [wx, wy] = makeView(cv, S.viewRange).inv(sx, sy);
    const boxes = (S.loaded.header.boxes || []).map(normalizeBox);
    const hit = pickBoxAt(wx, wy, boxes);
    if (hit >= 0) {
      S.targetIndex = hit;
      updateTargetInfo();
      if (S.attack !== "remove") {
        S.attack = "remove";
        $("atkRemove").click();
        return;
      }
    } else if (S.overlay) {
      S.spooferPos = [wx, wy, S.spooferPos[2]];
    }
    recompute();
  };

  $("upload").onchange = async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    try {
      const buf = await file.arrayBuffer();
      S.loaded = file.name.endsWith(".atlasframe") ? parseAtlasFrame(buf) : parseRawLidar(buf, { name: file.name });
      S.sceneSlug = "upload:" + file.name;
      S.frame = null;
      if (SENSORS[S.loaded.header.sensor]) {
        S.sensorId = S.loaded.header.sensor;
        $("sensor").value = S.sensorId;
      }
      ensureFrame();
      autoTarget(false);
      labels();
      await recompute();
      setStatus(`Loaded ${file.name} — ${S.frame.count.toLocaleString()} points (no ground truth boxes)`, "ok");
    } catch (e) {
      setStatus("Upload failed: " + e.message, "err");
    }
  };

  $("exportBtn").onclick = () => {
    if (!S.last || !S.last.result) return;
    const { keepMask, spoof, result, phantomBox } = S.last;
    const cloud = composeCloud({ frame: S.frame, keepMask, spoof });
    const spoofGt = S.attack === "inject" && phantomBox && spoof && spoof.x.length
      ? [...phantomBox.center, ...phantomBox.size, phantomBox.yaw]
      : targetBox();
    const bundle = buildAttackBundle({
      cloud,
      sensor: S.last.sensor,
      frameId: S.loaded.header.frameId ?? S.sceneSlug,
      dataset: S.loaded.header.dataset,
      sceneSlug: S.sceneSlug,
      spoofGt,
      attack: {
        mode: S.attack, difficulty: S.difficulty, seed: S.seed,
        azWidthDeg: S.azWidthDeg, pRemove: S.pRemove,
        intensityMode: S.intensityMode, atlasFaithful: !S.overlay,
        targetIndex: S.targetIndex,
      },
      spooferConfig: S.last.spoofer,
      stats: result.stats || result.meta,
      scheduledAttack: true,
    });
    downloadBundle(bundle, bundleFilename({ frameId: bundle.header.frameId, mode: S.attack, slug: S.sceneSlug }));
    setStatus(`Exported ${bundle.header.nPoints.toLocaleString()} points — convert with tools/import_attack.py`, "ok");
  };

  // Optional bootstrap from the URL, so a configuration is shareable and can be
  // driven headlessly: ?mode=remove&scene=kitti-parked-row&difficulty=hard
  const q = new URLSearchParams(location.search);
  const scene = q.get("scene");
  const initial = S.manifest.find((m) => m.slug === scene) ? scene : S.manifest[0].slug;
  if (q.has("difficulty") && ["easy", "medium", "hard"].includes(q.get("difficulty"))) {
    S.difficulty = q.get("difficulty");
    $(["dEasy", "dMedium", "dHard"][["easy", "medium", "hard"].indexOf(S.difficulty)])
      .setAttribute("aria-pressed", "true");
  }
  if (q.get("azWidth")) {
    S.azWidthDeg = parseFloat(q.get("azWidth"));
    $("azWidth").value = S.azWidthDeg;
  }
  if (q.get("forceActive") === "1") {
    S.forceActive = true;
    $("forceActive").checked = true;
  }
  if (q.get("overlay") === "1") {
    S.overlay = true;
    $("overlay").checked = true;
    $("spooferRow").style.display = "";
  }
  if (q.get("spin") === "0") {
    S.spinning = false;
    $("fullBtn").click();
  }

  $("scene").value = initial;
  labels();
  await selectScene(initial);

  if (q.get("mode") === "remove") {
    S.attack = "remove";
    $("atkInject").setAttribute("aria-pressed", "false");
    $("atkRemove").setAttribute("aria-pressed", "true");
    $("injectOpts").style.display = "none";
    $("removeOpts").style.display = "";
    if (q.get("target")) {
      S.targetIndex = parseInt(q.get("target"), 10);
      updateTargetInfo();
    }
    await recompute();
  }

  labels();
  requestAnimationFrame(loop);
}
