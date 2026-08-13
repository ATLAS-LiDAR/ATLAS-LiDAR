/**
 * Application orchestration.
 *
 * Two caches keep this interactive on 120k-point frames: the loaded cloud is
 * cached per scene, and the lattice index (PointFrame) is cached per
 * scene+sensor configuration. Only the attack is recomputed when a spoofer knob
 * moves, and that costs O(reachable cells), not O(points).
 */
import { configureSensor, listSensors, SENSORS } from "../core/sensor.js";
import { configureSpoofer, listSpoofers, SPOOFERS, reachableCells } from "../core/spoofer.js";
import { injectPhantom } from "../core/inject.js";
import { removeReturns } from "../core/remove.js";
import { makePhantom } from "../core/phantom.js";
import { analyzeAttack, classifyRealizability, firingBudget } from "../core/realizability.js";
import { syncHitProbability, minRelayRange } from "../core/tof.js";
import { loadAtlasFrame, loadManifest, parseAtlasFrame, parseRawLidar, toPointFrame } from "../io/atlasframe.js";
import { renderBEV, renderRangeImage, makeView } from "./render.js";
import { composeCloud } from "../core/compose.js";
import { buildAttackBundle, downloadBundle, bundleFilename } from "../io/export.js";

const DEG = Math.PI / 180;
const $ = (id) => document.getElementById(id);

const S = {
  manifest: [],
  sceneSlug: null,
  loaded: null,
  frame: null,
  frameKey: null,
  sensorId: "hdl64e",
  hz: 10,
  returnMode: "strongest",
  spooferId: "cao2023",
  power: null,
  divergenceMrad: 3,
  latencyNs: 10,
  prfHz: 100000,
  jitterNs: 1,
  syncMode: "locked",
  canPredict: true,
  attack: "inject",
  shape: "box",
  phantomRange: 30,
  phantomLen: 4.2,
  intensityMode: "range-conditioned",
  attackerRange: 0.2,
  groundAnchor: true,
  spooferPos: [25, 6, 0],
  viewRange: 80,
  showRemoved: true,
  showBoxes: true,
  seed: 7,
};

function setStatus(msg, cls = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

function sensor() {
  return configureSensor(S.sensorId, { hz: S.hz, returnMode: S.returnMode });
}

function spoofer() {
  const base = SPOOFERS[S.spooferId];
  return configureSpoofer(S.spooferId, {
    powerRatio: S.power ?? base.powerRatio,
    divergenceMrad: S.divergenceMrad,
    latencyS: S.latencyNs * 1e-9,
    prfHz: S.prfHz,
    jitterS: S.jitterNs * 1e-9,
    syncMode: S.syncMode,
    canPredict: S.canPredict,
  });
}

/* ------------------------------ data loading ------------------------------ */

async function selectScene(slug) {
  const entry = S.manifest.find((m) => m.slug === slug);
  if (!entry) return;
  setStatus("Loading " + entry.label + "…");
  const t0 = performance.now();
  S.loaded = await loadAtlasFrame("assets/frames/" + entry.file);
  S.sceneSlug = slug;
  // Adopt the sensor the frame was actually recorded with.
  if (SENSORS[S.loaded.header.sensor]) {
    S.sensorId = S.loaded.header.sensor;
    $("sensor").value = S.sensorId;
  }
  S.frame = null;
  ensureFrame();
  setStatus(
    `${entry.label} — ${S.frame.count.toLocaleString()} points indexed in ${(performance.now() - t0).toFixed(0)} ms`,
    "ok"
  );
  recompute();
}

function ensureFrame() {
  const sn = sensor();
  const key = `${S.sceneSlug}|${sn.id}|${sn.hz}|${sn.returnMode}`;
  if (S.frame && S.frameKey === key) return S.frame;
  const t0 = performance.now();
  S.frame = toPointFrame(S.loaded, sn);
  S.frameKey = key;
  S.indexMs = performance.now() - t0;
  return S.frame;
}

/* ------------------------------- compute --------------------------------- */

function recompute() {
  if (!S.loaded) return;
  const sn = ensureFrame().sensor;
  const frame = S.frame;
  const sp = spoofer();

  const t0 = performance.now();
  const reach = reachableCells({ frame, sensor: sn, spoofer: sp, positionXYZ: S.spooferPos });

  let result = null;
  let spoofPts = null;
  let keepMask = null;
  let replacement = null;

  if (S.attack === "inject") {
    const bearing = reach.geometry.bearing;
    const box = {
      center: [
        Math.cos(bearing) * S.phantomRange,
        Math.sin(bearing) * S.phantomRange,
        -1.2,
      ],
      size: [S.phantomLen, Math.max(1.4, S.phantomLen * 0.45), 1.6],
      yaw: bearing + Math.PI / 2,
    };
    const phantom = reach.feasible
      ? makePhantom({ shape: S.shape, sensor: sn, reach, box, range: S.phantomRange })
      : { x: new Float64Array(0), y: new Float64Array(0), z: new Float64Array(0) };

    result = injectPhantom({
      frame,
      sensor: sn,
      spoofer: sp,
      positionXYZ: S.spooferPos,
      phantom,
      reach,
      dataset: S.loaded.header.dataset,
      intensityMode: S.intensityMode,
      constantIntensity: 100,
      seed: S.seed,
      groundAnchor: S.groundAnchor,
    });
    spoofPts = result.spoof;
    keepMask = result.keepMask;
  } else {
    result = removeReturns({
      frame,
      sensor: sn,
      spoofer: sp,
      positionXYZ: S.spooferPos,
      attackerRangeM: S.attackerRange,
      reach,
      seed: S.seed,
    });
    keepMask = result.keepMask;
    replacement = result.replacement;
  }
  const computeMs = performance.now() - t0;

  // remember what produced the current view, so Export ships exactly this
  S.last = { sensor: sn, spoofer: sp, reach, keepMask, spoof: spoofPts, replacement, result };

  render(frame, sn, reach, keepMask, spoofPts, replacement);
  updateReadouts(sn, sp, reach, result, computeMs);
}

/* -------------------------------- render --------------------------------- */

function render(frame, sn, reach, keepMask, spoofPts, replacement) {
  const boxes = S.showBoxes ? S.loaded.header.boxes || [] : [];
  const common = { frame, boxes, reach, spooferPos: S.spooferPos, rangeM: S.viewRange };
  renderBEV($("cvClean"), { ...common, keepMask: null, spoof: null, showRemoved: false });
  renderBEV($("cvPert"), {
    ...common,
    keepMask,
    spoof: spoofPts,
    replacement,
    showRemoved: S.showRemoved,
  });
  renderRangeImage($("cvLattice"), {
    frame,
    sensor: sn,
    keepMask,
    reach,
    spoofCells: spoofPts ? spoofPts.cell : null,
  });
}

function fmt(n, d = 0) {
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}

function updateReadouts(sn, sp, reach, result, computeMs) {
  const st = result.stats;
  $("stPoints").textContent = S.frame.count.toLocaleString();
  $("stCells").textContent = reach.cells.length.toLocaleString();
  $("stEnv").innerHTML = reach.envelope.feasible
    ? fmt((2 * reach.envelope.halfWidthAz) / DEG, 1) + "<small>°</small>"
    : "—";
  $("stStandoff").innerHTML = fmt(reach.geometry.standoff, 1) + "<small>m</small>";

  const floor = minRelayRange({ standoffM: reach.geometry.standoff, latencyS: sp.latencyS });
  $("stFloor").innerHTML = fmt(floor, 1) + "<small>m</small>";

  const pSync = syncHitProbability({
    sensor: sn,
    mode: sp.syncMode,
    jitterS: sp.jitterS,
    toleranceM: 0.5,
  });
  $("stSync").innerHTML = fmt(pSync * 100, 1) + "<small>%</small>";

  const budget = firingBudget({ spoofer: sp, sensor: sn, requiredCells: reach.cells.length });
  $("stBudget").innerHTML =
    budget.maxCellsPerRevolution.toLocaleString() + "<small> pts/rev</small>";
  $("stBudget").parentElement.className = "stat" + (budget.withinBudget ? "" : " warn2");

  const injected = S.attack === "inject" ? st.injected : 0;
  const destroyed = S.attack === "inject" ? st.suppressed : st.removed;
  $("stInject").textContent = injected.toLocaleString();
  $("stRemoved").textContent = destroyed.toLocaleString();

  const analysis = analyzeAttack({
    reach,
    sensor: sn,
    nInjected: S.attack === "inject" ? st.injected : (st.removed + st.replaced),
  });
  $("stFill").textContent = fmt(analysis.fillUnique, 3);

  const verdict = classifyRealizability(analysis, {
    maxReturnsPerCell: sn.returnMode === "dual" ? 2 : 1,
  });
  $("verdict").className = "verdict " + verdict.verdict;
  $("vName").textContent = verdict.verdict.replace("-", " ").toUpperCase();
  $("vReasons").innerHTML = verdict.reasons.map((r) => "<li>" + r + "</li>").join("");

  const rj = st.rejected || {};
  $("rjReach").textContent = rj.unreachable ?? "—";
  $("rjResp").textContent = rj.response ?? "—";
  $("rjTof").textContent = rj.tof ?? "—";
  $("rjMot").textContent = rj.mot ?? "—";
  $("rjSync").textContent = rj.sync ?? "—";
  $("rjCell").textContent = rj.cellConflict ?? "—";
  $("rjOcc").textContent = rj.occluded ?? "—";

  // Status line: explain the binding constraint rather than just counting.
  if (!reach.lineOfSight.clear) {
    setStatus(
      `Line of sight blocked by real geometry at ${fmt(reach.lineOfSight.occluderRange, 1)} m — ` +
        `the spoofer cannot illuminate the receiver from behind that surface.`,
      "err"
    );
  } else if (!reach.envelope.feasible) {
    setStatus(
      "Received power is below the victim's detection threshold at this standoff and divergence — no cell is reachable.",
      "err"
    );
  } else if (S.attack === "remove" && st.outcome === "infeasible") {
    setStatus("Removal infeasible: " + st.reason, "err");
  } else if (S.attack === "inject" && st.injected === 0 && st.requested > 0) {
    const worst = Object.entries(rj).sort((a, b) => b[1] - a[1])[0];
    setStatus(
      `All ${st.requested} candidate points rejected — dominant cause: ${worst[0]} (${worst[1]}).`,
      "warn"
    );
  } else if (S.attack === "inject") {
    setStatus(
      `${st.injected} points injected across ${reach.cells.length} reachable cells, ` +
        `destroying ${st.suppressed} genuine returns` +
        (reach.firingLimited ? " · firing-budget limited" : "") +
        (reach.envelope.extrapolation ? " · envelope exceeds demonstrated 45°" : "") +
        ` · ${computeMs.toFixed(0)} ms`,
      reach.envelope.extrapolation ? "warn" : "ok"
    );
  } else {
    setStatus(
      `${st.outcome}: ${st.removed} returns destroyed, ${st.replaced} range-replaced · ` +
        `aim-point rate ${fmt(st.peakRate, 2)}, fringe ${fmt(st.edgeRate, 2)} · ${computeMs.toFixed(0)} ms`,
      "ok"
    );
  }
}

/* -------------------------------- wiring --------------------------------- */

function seg(aId, bId, onA, onB) {
  const a = $(aId);
  const b = $(bId);
  a.onclick = () => {
    a.setAttribute("aria-pressed", "true");
    b.setAttribute("aria-pressed", "false");
    onA();
  };
  b.onclick = () => {
    b.setAttribute("aria-pressed", "true");
    a.setAttribute("aria-pressed", "false");
    onB();
  };
}

function labels() {
  const sp = spoofer();
  const sn = sensor();
  $("hzVal").textContent = S.hz + " Hz";
  $("sensorSpec").textContent = `${sn.nBeams}b · ${(sn.azRes / DEG).toFixed(3)}°`;
  $("powVal").textContent = "×" + (S.power ?? sp.powerRatio).toFixed(1);
  $("divVal").textContent = S.divergenceMrad.toFixed(1) + " mrad";
  $("latVal").textContent = S.latencyNs.toFixed(0) + " ns";
  $("prfVal").textContent = (S.prfHz / 1000).toFixed(0) + " kHz";
  $("jitVal").textContent =
    S.jitterNs < 1 ? (S.jitterNs * 1000).toFixed(0) + " ps" : S.jitterNs.toFixed(1) + " ns";
  $("prVal").textContent = S.phantomRange.toFixed(1) + " m";
  $("plVal").textContent = S.phantomLen.toFixed(1) + " m";
  $("arVal").textContent = S.attackerRange.toFixed(2) + " m";
  $("rangeVal").textContent = S.viewRange + " m";
  const e = S.manifest.find((m) => m.slug === S.sceneSlug);
  $("sceneSpec").textContent = e ? `${(e.nPoints / 1000).toFixed(0)}k pts` : "";
}

function bindRange(id, apply, transform = (v) => v) {
  $(id).oninput = (ev) => {
    apply(transform(parseFloat(ev.target.value)));
    labels();
    recompute();
  };
}

export async function boot() {
  // sensor + spoofer selects
  const sensorSel = $("sensor");
  for (const id of listSensors()) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = SENSORS[id].name;
    sensorSel.appendChild(o);
  }
  const spSel = $("spooferPreset");
  for (const id of listSpoofers()) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = SPOOFERS[id].name;
    spSel.appendChild(o);
  }

  S.manifest = await loadManifest("assets/frames/manifest.json");
  const sceneSel = $("scene");
  for (const m of S.manifest) {
    const o = document.createElement("option");
    o.value = m.slug;
    o.textContent = m.label;
    sceneSel.appendChild(o);
  }

  sceneSel.onchange = (e) => selectScene(e.target.value);
  sensorSel.onchange = (e) => {
    S.sensorId = e.target.value;
    S.frame = null;
    labels();
    recompute();
  };
  spSel.onchange = (e) => {
    S.spooferId = e.target.value;
    const b = SPOOFERS[S.spooferId];
    S.power = b.powerRatio;
    S.divergenceMrad = b.divergenceMrad;
    S.latencyNs = b.latencyS * 1e9;
    S.prfHz = b.prfHz;
    S.jitterNs = b.jitterS * 1e9;
    S.syncMode = b.syncMode;
    S.canPredict = b.canPredict;
    $("power").value = Math.log10(S.power);
    $("divergence").value = S.divergenceMrad;
    $("latency").value = S.latencyNs;
    $("prf").value = Math.log10(S.prfHz);
    $("jitter").value = Math.log10(S.jitterNs);
    $("canPredict").checked = S.canPredict;
    $(S.syncMode === "locked" ? "syncLocked" : "syncFree").click();
    labels();
    recompute();
  };

  seg("rmSingle", "rmDual",
    () => { S.returnMode = "strongest"; S.frame = null; recompute(); },
    () => { S.returnMode = "dual"; S.frame = null; recompute(); });
  seg("syncLocked", "syncFree",
    () => { S.syncMode = "locked"; recompute(); },
    () => { S.syncMode = "free"; recompute(); });
  seg("atkInject", "atkRemove",
    () => {
      S.attack = "inject";
      $("injectOpts").style.display = "";
      $("removeOpts").style.display = "none";
      recompute();
    },
    () => {
      S.attack = "remove";
      $("injectOpts").style.display = "none";
      $("removeOpts").style.display = "";
      recompute();
    });
  seg("shapeBox", "shapeWall",
    () => { S.shape = "box"; recompute(); },
    () => { S.shape = "wall"; recompute(); });

  bindRange("hz", (v) => { S.hz = v; S.frame = null; });
  bindRange("power", (v) => { S.power = Math.pow(10, v); });
  bindRange("divergence", (v) => { S.divergenceMrad = v; });
  bindRange("latency", (v) => { S.latencyNs = v; });
  bindRange("prf", (v) => { S.prfHz = Math.pow(10, v); });
  bindRange("jitter", (v) => { S.jitterNs = Math.pow(10, v); });
  bindRange("phantomRange", (v) => { S.phantomRange = v; });
  bindRange("phantomLen", (v) => { S.phantomLen = v; });
  bindRange("attackerRange", (v) => { S.attackerRange = v; });
  bindRange("viewRange", (v) => { S.viewRange = v; });

  $("intensityMode").onchange = (e) => { S.intensityMode = e.target.value; recompute(); };
  $("canPredict").onchange = (e) => { S.canPredict = e.target.checked; recompute(); };
  $("groundAnchor").onchange = (e) => { S.groundAnchor = e.target.checked; recompute(); };
  $("showRemoved").onchange = (e) => { S.showRemoved = e.target.checked; recompute(); };
  $("showBoxes").onchange = (e) => { S.showBoxes = e.target.checked; recompute(); };

  $("exportBtn").onclick = () => {
    if (!S.last) return;
    const { sensor: sn, spoofer: sp, reach, keepMask, spoof, replacement, result } = S.last;
    const cloud = composeCloud({ frame: S.frame, keepMask, spoof, replacement });

    // For an injection the phantom box is what a detector must match; derive it
    // from the points that actually landed, not the requested geometry.
    let spoofGt = null;
    if (S.attack === "inject" && spoof && spoof.x.length > 0) {
      let sx = 0, sy = 0, zMin = Infinity, zMax = -Infinity;
      for (let i = 0; i < spoof.x.length; i++) {
        sx += spoof.x[i];
        sy += spoof.y[i];
        zMin = Math.min(zMin, spoof.z[i]);
        zMax = Math.max(zMax, spoof.z[i]);
      }
      const n = spoof.x.length;
      const bearing = reach.geometry.bearing;
      spoofGt = [
        sx / n, sy / n, (zMin + zMax) / 2,
        S.phantomLen, Math.max(1.4, S.phantomLen * 0.45), 1.6, bearing + Math.PI / 2,
      ];
    }

    const bundle = buildAttackBundle({
      cloud,
      sensor: sn,
      frameId: S.loaded.header.frameId ?? S.sceneSlug,
      dataset: S.loaded.header.dataset,
      sceneSlug: S.sceneSlug,
      spoofGt,
      attack: {
        mode: S.attack, shape: S.shape, phantomRange: S.phantomRange,
        phantomLen: S.phantomLen, intensityMode: S.intensityMode,
        attackerRangeM: S.attackerRange, groundAnchor: S.groundAnchor,
        spooferPos: S.spooferPos, seed: S.seed, hz: S.hz, returnMode: S.returnMode,
      },
      spooferConfig: sp,
      stats: { ...result.stats, reachableCells: reach.cells.length, feasible: reach.feasible },
      scheduledAttack: true,
    });

    downloadBundle(bundle, bundleFilename({
      frameId: bundle.header.frameId, mode: S.attack, slug: S.sceneSlug,
    }));
    setStatus(
      `Exported ${bundle.header.nPoints.toLocaleString()} points ` +
        `(${(bundle.byteLength / 1024 / 1024).toFixed(2)} MB). ` +
        `Convert with tools/import_attack.py, then score with eval_matrix_kitti.py.`,
      "ok"
    );
  };

  // click to place the spoofer
  const cv = $("cvClean");
  cv.onclick = (ev) => {
    const r = cv.getBoundingClientRect();
    const sx = ((ev.clientX - r.left) * cv.width) / r.width;
    const sy = ((ev.clientY - r.top) * cv.height) / r.height;
    const [wx, wy] = makeView(cv, S.viewRange).inv(sx, sy);
    S.spooferPos = [wx, wy, S.spooferPos[2]];
    recompute();
  };

  // upload
  $("upload").onchange = async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    try {
      const buf = await file.arrayBuffer();
      S.loaded = file.name.endsWith(".atlasframe")
        ? parseAtlasFrame(buf)
        : parseRawLidar(buf, { name: file.name });
      S.sceneSlug = "upload:" + file.name;
      S.frame = null;
      if (SENSORS[S.loaded.header.sensor]) {
        S.sensorId = S.loaded.header.sensor;
        $("sensor").value = S.sensorId;
      }
      ensureFrame();
      labels();
      recompute();
      setStatus(
        `Loaded ${file.name} — ${S.frame.count.toLocaleString()} points, ` +
          `assumed ${S.loaded.header.dataset} / ${S.loaded.header.sensor}` +
          (S.loaded.header.stride ? ` (stride ${S.loaded.header.stride})` : ""),
        "ok"
      );
    } catch (e) {
      setStatus("Upload failed: " + e.message, "err");
    }
  };

  const base = SPOOFERS[S.spooferId];
  S.power = base.powerRatio;
  $("power").value = Math.log10(S.power);
  $("prf").value = Math.log10(S.prfHz);
  $("jitter").value = Math.log10(S.jitterNs);

  labels();
  await selectScene(S.manifest[0].slug);
  labels();
}
