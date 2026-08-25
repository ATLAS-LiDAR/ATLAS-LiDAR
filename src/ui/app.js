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
import { ahfrRemoval, ahfrAim, selectInitialTarget, selectNearestFrontCar,
         matchWithVelocity, makeAtlasRng, pRemoveForWidth, ATLAS_AZ_WIDTHS,
         NUSCENES_TRACK, AHFR_DEFAULTS, sectorMask } from "../core/atlas-removal.js";
import { placeTraceRelative, placeTraceGlobal, transformSpoofToEgoFrame } from "../core/atlas-inject.js";
import { loadSequence, loadSequenceManifest, frameOfSequence } from "../io/sequence.js";
import { buildAttackSchedule } from "../core/schedule.js";
import { loadTraceLibrary, loadTraceManifest, allTraces } from "../io/traces.js";
import { loadAtlasFrame, loadManifest, parseAtlasFrame, parseRawLidar, toPointFrame } from "../io/atlasframe.js";
import { composeCloud } from "../core/compose.js";
import { buildAttackBundle, downloadBundle, bundleFilename } from "../io/export.js";
import { pickBoxAt, normalizeBox } from "../core/iou.js";
import { attackerGeometry, worldFromEgoPoint, ATTACKER_MODES,
         victimApertureRad, attackerCanIlluminate, illuminationWindow, heardColumns,
         illuminationDuty } from "../core/attacker.js";
import { makeAccumulator, resetAccumulator, markSwept, accumulatedColumns,
         advanceSweep, sweepAzToCell, beamInSector,
         sectorDwellTime, pulsesDuringDwell, victimCellsInSector } from "../core/sweep.js";
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
  traceIdx: null,      // null = let ATLAS sample it
  phantomAt: null,     // null = let ATLAS sample the offset
  standoff: 18,
  laserPhase: 0,
  attackerMode: "static",   // roadside box vs vehicle-mounted
  attackerWorld: null,      // fixed world position of a roadside attacker
  attackerGeom: null,
  // false: show every point ATLAS generated. true: dim the ones this attacker
  // could not have produced. Neither changes what is generated.
  gateByReach: false,
  // sequence mode
  isSequence: false,
  seq: null,
  seqManifest: [],
  frameIndex: 0,
  placement: "relative",   // relative = follows ego, global = world-fixed
  schedMode: "always",
  schedule: null,
  globalPhantom: null,     // {points, box} in world coords, placed once
  track: null,             // per-keyframe {targetBox, incomingState}
  // Accumulation: the display starts empty and each azimuth sector is filled in
  // as the beam passes it, keeping its last observation thereafter.
  accum: null,
  layers: new Map(),       // observation id -> {frame, keepMask, spoof, replacement}
  layerId: 0,
  azWidthDeg: AHFR_DEFAULTS.azWidthDeg,
  pRemove: AHFR_DEFAULTS.pRemove,
  pRemoveOverride: false,
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
/** Which dataset's ATLAS constants apply to the loaded frame. */
const datasetOf = () => (S.loaded?.header.dataset === "kitti" ? "kitti" : "nuscenes");
/** p_remove is derived from az_width unless explicitly overridden. */
const effectivePRemove = () => (S.pRemoveOverride ? S.pRemove : pRemoveForWidth(S.azWidthDeg));
const spoofer = () => configureSpoofer(S.spooferId);

/** GT boxes of the current frame, as ATLAS class-tagged vectors. */
function atlasBoxes() {
  return atlasBoxesOf(S.loaded?.header.boxes || []);
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
  const seqEntry = S.seqManifest.find((m) => m.slug === slug);
  if (seqEntry) return selectSequence(seqEntry);

  const entry = S.manifest.find((m) => m.slug === slug);
  if (!entry) return;
  setStatus("Loading " + entry.label + "…");
  S.isSequence = false;
  S.seq = null;
  S.schedule = null;
  S.globalPhantom = null;
  S.track = null;
  $("seqOpts").style.display = "none";
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
  clearAccumulation();
  S.sweepAz = 0;
}

/** Load a driving sequence and prepare its schedule, phantom and track. */
async function selectSequence(entry) {
  setStatus("Loading " + entry.label + "…");
  S.seq = await loadSequence("assets/sequences/" + entry.file);
  S.isSequence = true;
  S.sceneSlug = entry.slug;
  S.frameIndex = 0;
  $("seqOpts").style.display = "";
  $("frameIdx").max = String(S.seq.header.nFrames - 1);
  $("frameIdx").value = "0";
  S.sensorId = S.seq.header.sensor;
  $("sensor").value = S.sensorId;
  await loadKeyframe(0);
  await traceLib();
  buildSchedule();
  await prepareSequenceAttack();
  labels();
  await recompute();
  clearAccumulation();
  S.sweepAz = 0;
}

async function loadKeyframe(k) {
  S.frameIndex = Math.max(0, Math.min(S.seq.header.nFrames - 1, k));
  S.loaded = frameOfSequence(S.seq, S.frameIndex);
  S.frame = null;
  ensureFrame();
}

function buildSchedule() {
  if (!S.isSequence || S.schedMode === "always") {
    S.schedule = null;
    return;
  }
  S.schedule = buildAttackSchedule({
    nFrames: S.seq.header.nFrames,
    mode: S.schedMode,
    seed: S.seed,
    label: S.sceneSlug,
  });
}

const frameAttacked = () =>
  !S.isSequence || !S.schedule ? true : S.schedule.isAttacked(S.frameIndex);

/**
 * Precompute what persists across keyframes:
 *  - a world-fixed phantom, placed once in the anchor frame then projected down
 *  - the removal target's track, plus the aim state entering each frame
 * Both are cheap (boxes only, no point work), so scrubbing stays instant.
 */
async function prepareSequenceAttack() {
  if (!S.isSequence) return;
  const frames = S.seq.header.frames;
  // ATLAS anchors a world-fixed phantom on the LAST frame of the attack span
  // (`anchor_pose = idx_to_pose[max(span)]`), not the first.
  const attacked = S.schedule ? S.schedule.attacked : null;
  const anchor = attacked && attacked.length
    ? attacked[attacked.length - 1]
    : frames.length - 1;

  // world-fixed phantom, placed in the anchor keyframe
  if (S.attack === "inject" && S.placement === "global") {
    const { traces } = await traceLib();
    S.globalPhantom = placeTraceGlobal(traces, {
      seed: S.seed,
      traceIdx: S.traceIdx,
      offsetXY: S.phantomAt,
      dataset: datasetOf(),
      anchorPose: frames[anchor].pose,
    });
  } else {
    S.globalPhantom = null;
  }

  // target track + aim state, frame by frame
  if (S.attack === "remove") {
    const dt = S.seq.header.dtSec;
    const track = [];
    let prev = null;
    let state = null;
    let lost = false;
    for (let k = 0; k < frames.length; k++) {
      const boxes = atlasBoxesOf(frames[k].boxes);
      let targetBox = null;
      if (lost) {
        // ATLAS's track_sequence does `if new_box is None: break` - once the
        // target is lost it gives up for the rest of the sequence rather than
        // re-acquiring a different vehicle.
        targetBox = null;
      } else if (k === 0) {
        const chosen = S.targetIndex >= 0 && boxes[S.targetIndex]
          ? boxes[S.targetIndex]
          : selectInitialTarget(boxes, {
              desiredRange: NUSCENES_TRACK.desiredRange, tol: AHFR_DEFAULTS.tol,
              targetClass: 1, minRange: NUSCENES_TRACK.minRange,
              negative: NUSCENES_TRACK.negative,
            }).box;
        targetBox = chosen || null;
        if (!targetBox) lost = true;
      } else {
        targetBox = matchWithVelocity(prev, null, boxes, {
          dt, maxStep: NUSCENES_TRACK.maxStep,
        }).box;
        if (!targetBox) lost = true;
      }
      const incoming = state;
      if (targetBox) {
        const aim = ahfrAim({ targetBox, state, rng: makeAtlasRng(S.seed + k) });
        state = aim.state;
      }
      track.push({ targetBox, incomingState: incoming, lost });
      if (targetBox) prev = targetBox;
    }
    S.track = track;
  } else {
    S.track = null;
  }
}

/** GT boxes -> ATLAS 10-vectors [x,y,z,dx,dy,dz,yaw,vx,vy,class]. */
function atlasBoxesOf(boxes) {
  return boxes.map((b) => [
    b.center[0], b.center[1], b.center[2],
    b.size[0], b.size[1], b.size[2], b.yaw,
    b.velocity ? b.velocity[0] : 0, b.velocity ? b.velocity[1] : 0,
    labelToClass(b.label),
  ]);
}

/** ATLAS's own target choice. */
function autoTarget(announce = true) {
  // The two generators pick differently, so follow whichever applies.
  // KITTI (single frames): nearest car in front beyond 10 m, by footprint range.
  // nuScenes (sequences): select_initial_target at 40 m, min 10 m, negative=True.
  const boxes = atlasBoxes();
  const { index } = datasetOf() === "kitti" && !S.isSequence
    ? selectNearestFrontCar(boxes, { minX: 10.0, targetClass: 1 })
    : selectInitialTarget(boxes, {
        desiredRange: NUSCENES_TRACK.desiredRange,
        tol: AHFR_DEFAULTS.tol,
        targetClass: 1,
        minRange: NUSCENES_TRACK.minRange,
        negative: NUSCENES_TRACK.negative,
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
  // ATLAS's A-HFR removal deletes returns rather than moving them, so this stays
  // null there; the Cao-mechanism model in remove.js is what produces
  // range-replaced points when the physics overlay drives removal.
  let replacement = null;
  let phantomBox = null;
  let atlasSector = null;
  let secMask = null;
  let centerAz = 0;

  const attackedNow = frameAttacked();

  if (S.attack === "inject") {
    const { traces } = await traceLib();
    let placed;
    if (S.isSequence && S.placement === "global" && S.globalPhantom) {
      // world-fixed: project the once-placed phantom into THIS keyframe
      const ego = transformSpoofToEgoFrame(
        S.globalPhantom.points, S.globalPhantom.box, S.loaded.header.pose
      );
      placed = { points: ego.points, box: ego.box, anno: { ...S.globalPhantom.anno } };
    } else {
      placed = placeTraceRelative(traces, {
        seed: S.seed,
        traceIdx: S.traceIdx,
        offsetXY: S.phantomAt,
      });
    }
    if (!attackedNow) {
      placed = { ...placed, points: { x: new Float64Array(0), y: new Float64Array(0), z: new Float64Array(0), intensity: new Float32Array(0) } };
    }
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

    // The attack window for an injection is the azimuth span the phantom's own
    // points occupy - that is the set of victim rays the attacker has to hit.
    let spanMin = 0;
    let spanMax = 0;
    for (let i = 0; i < spoof.az.length; i++) {
      let d = spoof.az[i] - centerAz;
      if (d > Math.PI) d -= 2 * Math.PI;
      if (d < -Math.PI) d += 2 * Math.PI;
      spanMin = Math.min(spanMin, d);
      spanMax = Math.max(spanMax, d);
    }
    const spanRad = Math.max(spanMax - spanMin, sn.azRes);
    atlasSector = { azCenter: centerAz, azWidthDeg: (spanRad * 180) / Math.PI };
    result.elWidthDeg = 16;
  } else {
    const tracked = S.isSequence && S.track ? S.track[S.frameIndex] : null;
    const tb = tracked ? tracked.targetBox : targetBox();
    if (!tb) {
      setStatus("Pick a vehicle to hide — click one in the left view, or use auto-select.", "warn");
      S.last = { sensor: sn, reach, keepMask: null, spoof: null, result: null, phantomBox: null, atlasSector: null, secMask: null, centerAz: 0 };
      draw();
      return;
    }
    result = attackedNow
      ? ahfrRemoval({
          frame,
          targetBox: tb,
          azWidthDeg: S.azWidthDeg,
          pRemove: effectivePRemove(),
          forceActive: S.forceActive,
          state: tracked ? tracked.incomingState : null,
          rng: makeAtlasRng(S.seed + S.frameIndex),
        })
      : ahfrRemoval({
          frame, targetBox: tb, azWidthDeg: S.azWidthDeg, pRemove: 0,
          forceActive: false, state: tracked ? tracked.incomingState : null,
          rng: makeAtlasRng(S.seed + S.frameIndex),
        });
    if (!attackedNow) result.meta.inactiveReason = "schedule_gap";
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

  // Where the hardware actually stands. A roadside attacker holds a world
  // position, so as the ego drives past its bearing and range both change and its
  // aim drifts off the attacked sector - which is physically real, and judged
  // against its reachable envelope rather than against exact alignment.
  const sectorWidthRad = atlasSector ? atlasSector.azWidthDeg * DEG : 0;
  if (S.attackerMode === "static" && !S.attackerWorld) {
    // default: standing where the attack is aimed, in world coordinates
    S.attackerWorld = worldFromEgoPoint(
      [Math.cos(centerAz) * S.standoff, Math.sin(centerAz) * S.standoff, 0],
      S.loaded.header.pose || null
    );
  }
  const geom = attackerGeometry({
    mode: S.attackerMode,
    worldPos: S.attackerWorld,
    egoOffset: [Math.cos(centerAz) * S.standoff, Math.sin(centerAz) * S.standoff, 0],
    pose: S.loaded.header.pose || null,
    sensor: sn,
    spoofer: sp,
    sectorCenterAz: centerAz,
    sectorWidthRad,
  });
  S.attackerGeom = geom;
  const laserPos = [geom.egoPos[0], geom.egoPos[1]];

  // Which columns this attacker could act on at all. A spoofed return outside the
  // band is not producible from where it stands, however hard it fires.
  const apRadNow = victimApertureRad(sn);
  const bandNow = illuminationWindow({
    attackerBearing: geom.bearing,
    envelopeHalfWidthRad: geom.envelope.feasible ? geom.envelope.halfWidthAz : 0,
    apertureRad: apRadNow,
  });
  const heard = heardColumns(sn, bandNow);
  let producible = 0;
  let unproducible = 0;
  if (spoof) {
    for (let i = 0; i < spoof.az.length; i++) {
      const c = Math.min(sn.azCellCount - 1, Math.floor(spoof.az[i] / sn.azRes));
      if (heard[c]) producible++;
      else unproducible++;
    }
  }
  S.producible = producible;
  S.unproducible = unproducible;
  S.last = {
    sensor: sn, spoofer: sp, reach, keepMask, spoof, result,
    phantomBox, atlasSector, secMask, centerAz, laserPos,
    frame, replacement, heard,
    ms: performance.now() - t0,
  };

  // A new observation the beam can write into sectors from here on. Older ones
  // stay alive only while some sector still shows them.
  if (!S.accum || S.accum.azCellCount !== sn.azCellCount) S.accum = makeAccumulator(sn);
  S.layerId += 1;
  S.layers.set(S.layerId, { frame, keepMask, spoof, replacement, heard });
  pruneLayers();
  draw();
  report();
}

/** Drop observations no sector shows any more, so memory does not grow. */
function pruneLayers() {
  if (!S.accum) return;
  const live = new Set([S.layerId]);
  for (let c = 0; c < S.accum.observed.length; c++) {
    const o = S.accum.observed[c];
    if (o >= 0) live.add(o);
  }
  for (const id of [...S.layers.keys()]) if (!live.has(id)) S.layers.delete(id);
}

/** Start the display over from empty (new scene, or a manual frame jump). */
function clearAccumulation() {
  if (S.accum) resetAccumulator(S.accum);
  S.layers.clear();
  if (S.last) S.layers.set(S.layerId, {
    frame: S.last.frame, keepMask: S.last.keepMask,
    spoof: S.last.spoof, replacement: S.last.replacement,
  });
}

/* -------------------------------- render --------------------------------- */

function draw() {
  if (!S.last || !S.frame) return;
  const { sensor: sn, reach, keepMask, spoof, replacement, phantomBox, atlasSector, secMask, centerAz } = S.last;
  const frame = S.frame;

  const azIdx = sweepAzToCell(S.sweepAz, sn);
  const sweep = S.spinning ? { az: S.sweepAz, azIdx } : null;
  // Spinning: draw each sector from whatever observation owns it, so the view
  // builds up from empty and then holds. Paused: show the whole frame.
  const accum = S.spinning ? S.accum : null;

  const sectorActive = atlasSector
    ? beamInSector(S.sweepAz, atlasSector.azCenter, atlasSector.azWidthDeg * DEG)
    : false;

  // Whether the attacker can be heard right now depends on ITS bearing and its
  // reachable band, not on where the attack is aimed.
  const g = S.attackerGeom;
  const apRad = victimApertureRad(sn);
  const illum = g
    ? illuminationWindow({
        attackerBearing: g.bearing,
        envelopeHalfWidthRad: g.envelope.feasible ? g.envelope.halfWidthAz : 0,
        apertureRad: apRad,
      })
    : null;
  const firing = g && S.spinning
    ? attackerCanIlluminate({
        sweepAz: S.sweepAz,
        attackerBearing: g.bearing,
        envelopeHalfWidthRad: g.envelope.feasible ? g.envelope.halfWidthAz : 0,
        apertureRad: apRad,
      })
    : !!g;
  S.firing = firing;

  const laser = S.last.laserPos
    ? { pos: S.last.laserPos, active: firing, phase: S.laserPhase }
    : null;

  const boxes = S.showBoxes ? S.loaded.header.boxes || [] : [];
  const tb = targetBox();
  const targetView = tb && S.attack === "remove"
    ? { center: tb.slice(0, 3), size: tb.slice(3, 6), yaw: tb[6] }
    : null;

  const common = { frame, boxes, reach, spooferPos: S.overlay ? S.spooferPos : null, rangeM: S.viewRange, sweep };
  // the clean view accumulates the same sectors, but with no attack applied
  const cleanLayers = new Map();
  for (const [id, l] of S.layers) cleanLayers.set(id, { frame: l.frame });
  renderBEV($("cvClean"), {
    ...common,
    keepMask: null, spoof: null, showRemoved: false, targetBox: targetView,
    accum, layers: cleanLayers,
  });
  renderBEV($("cvPert"), {
    ...common,
    keepMask,
    spoof,
    replacement,
    atlasSector,
    sectorActive,
    laser,
    heard: S.gateByReach ? S.last.heard : null,
    aperture: sweep ? { az: S.sweepAz, halfWidthRad: apRad / 2 } : null,
    illumWindow: illum,
    targetBox: targetView,
    phantomBox,
    showRemoved: S.showRemoved,
    accum, layers: S.layers,
  });
  updateLaserInfo();
  updateSeqInfo();
  const secEl = $("stSectors");
  if (secEl && S.accum) {
    const obs = accumulatedColumns(S.accum);
    const tot = S.accum.observed.length;
    secEl.innerHTML = S.spinning
      ? `${obs.toLocaleString()}<small>/${tot.toLocaleString()}</small>`
      : "all";
    $("stSectorsWrap").className = "stat" + (S.spinning && obs < tot ? " warn2" : " good");
  }
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

/** The attacker's firing rate against the victim's scan, in plain numbers. */
function updateLaserInfo() {
  const el = $("laserInfo");
  if (!S.last || !S.last.atlasSector) {
    el.innerHTML = "—";
    return;
  }
  const sn = S.last.sensor;
  const sec = S.last.atlasSector;
  const widthRad = sec.azWidthDeg * DEG;
  const elWidthRad = (S.attack === "remove" ? S.last.result?.meta?.elWidthDeg ?? 16 : 16) * DEG;
  const dwell = sectorDwellTime(sn, widthRad);
  const prf = S.last.spoofer.prfHz;
  const pulses = pulsesDuringDwell(prf, dwell);
  const cells = victimCellsInSector(sn, widthRad, elWidthRad);
  const g = S.attackerGeom;
  const firing = S.firing ?? false;
  const apDeg = (victimApertureRad(sn) * 180) / Math.PI;
  const duty = g
    ? illuminationDuty({
        envelopeHalfWidthRad: g.envelope.feasible ? g.envelope.halfWidthAz : 0,
        apertureRad: victimApertureRad(sn),
      })
    : 0;
  const cov = g ? g.coverage : null;
  const covPct = cov ? (cov.fraction * 100).toFixed(0) : "—";
  const covClass = !cov ? "" : cov.withinEnvelope ? "" : cov.partial ? "warn" : "err";
  el.innerHTML =
    `${firing ? "<b>IN APERTURE — heard</b>" : "shut out"} · ` +
    `victim aperture ${apDeg.toFixed(2)}° · heard ${(duty * 100).toFixed(1)}% of each turn<br>` +
    `${S.attackerMode === "static" ? "roadside" : "vehicle-mounted"}` +
    ` at ${g ? g.standoff.toFixed(1) : "?"} m, bearing ${g ? g.bearingDeg.toFixed(0) : "?"}°<br>` +
    `reach ±${g ? g.envelopeHalfWidthDeg.toFixed(1) : "?"}° · attack ${sec.azWidthDeg.toFixed(1)}° ` +
    `at ${cov ? cov.offsetDeg.toFixed(0) : "?"}° off-axis → ` +
    `<span style="color:var(--${covClass === "err" ? "err" : covClass === "warn" ? "warn" : "ok"})">` +
    `${covPct}% reachable</span><br>` +
    `beam crosses it in ${(dwell * 1000).toFixed(2)} ms · ${pulses.toLocaleString()} pulses @ ` +
    `${(prf / 1000).toFixed(0)} kHz vs ${cells.toLocaleString()} cells`;
}

/** Where we are in the sequence, and what the schedule says about it. */
function updateSeqInfo() {
  if (!S.isSequence) return;
  const h = S.seq.header;
  const t = (S.frameIndex * h.dtSec).toFixed(1);
  const sched = S.schedule
    ? `${S.schedule.meta.mode} · L=${S.schedule.meta.attackLenL} of ${h.nFrames}` +
      (S.schedule.meta.nGaps ? ` · ${S.schedule.meta.nGaps} gaps` : "") +
      (S.schedule.meta.degenerate ? " · degenerate" : "")
    : "every keyframe";
  const on = frameAttacked();
  $("seqInfo").innerHTML =
    `frame ${S.frameIndex + 1}/${h.nFrames} · t=${t}s · ${h.dtSec.toFixed(2)}s apart (${h.hzKeyframe} Hz)<br>` +
    `${on ? "<b>ATTACK ON</b>" : "clean frame"} · ${sched}<br>` +
    `phantom ${S.placement === "global" ? "world-fixed" : "follows ego"}` +
    (S.attack === "remove" && S.track
      ? ` · track ${S.track[S.frameIndex].targetBox ? "held" : "LOST"}`
      : "");
  $("schedSpec").textContent = S.schedule
    ? `${S.schedule.meta.attackLenL}/${h.nFrames} frames`
    : `${h.nFrames} frames`;
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
  const obs = S.accum ? accumulatedColumns(S.accum) : 0;
  const total = S.accum ? S.accum.observed.length : 0;
  const rows = [
    stat("Frame points", S.frame.count.toLocaleString()),
    `<div class="stat" id="stSectorsWrap"><div class="k2">Sectors observed</div>` +
      `<div class="v" id="stSectors">${S.spinning ? obs.toLocaleString() + "<small>/" + total.toLocaleString() + "</small>" : "all"}</div></div>`,
  ];

  if (S.attack === "inject") {
    const st = result.stats;
    const p = result.placed;
    rows.push(
      stat("Phantom points", st.injected.toLocaleString()),
      stat(
        "Producible here",
        `${(S.producible ?? 0).toLocaleString()}<small>/${st.injected.toLocaleString()}</small>`,
        (S.unproducible ?? 0) > 0 ? "warn2" : "good"
      ),
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
  if (S.spinning && S.frame && S.last && !S.frozen) {
    const prevAz = S.sweepAz;
    S.sweepAz = advanceSweep(S.sweepAz, dt, S.last.sensor, S.speed);
    S.laserPhase = (S.laserPhase + dt * 3) % 1;
    // the beam observes every sector it crossed on this step
    if (S.accum) markSwept(S.accum, prevAz, S.sweepAz, S.layerId, S.last.sensor);
    // One displayed revolution advances one keyframe. nuScenes keyframes are 2 Hz
    // samples of a 20 Hz spin, so this compresses the gap between them - it keeps
    // the two clocks legible rather than pretending to be real time.
    if (S.isSequence && S.sweepAz < prevAz && !S.seqBusy) {
      S.seqBusy = true;
      advanceKeyframe(1).finally(() => { S.seqBusy = false; });
    }
    draw();
  }
  requestAnimationFrame(loop);
}

async function advanceKeyframe(delta) {
  if (!S.isSequence) return;
  const next = (S.frameIndex + delta + S.seq.header.nFrames) % S.seq.header.nFrames;
  await loadKeyframe(next);
  $("frameIdx").value = String(next);
  await recompute();
}

/* -------------------------------- wiring --------------------------------- */

function labels() {
  const sn = sensor();
  $("hzVal").textContent = S.hz + " Hz";
  $("sensorSpec").textContent = `${sn.nBeams}b · ${(sn.azRes / DEG).toFixed(3)}°`;
  $("azwVal").textContent = S.azWidthDeg + "°";
  $("prVal").textContent =
    effectivePRemove().toFixed(2) + (S.pRemoveOverride ? " (override)" : " (from az_width)");
  $("azwVal").textContent = S.azWidthDeg + "°";
  $("speedVal").textContent = S.speed.toFixed(2) + "×";
  $("rangeVal").textContent = S.viewRange + " m";
  $("standoffVal").textContent = S.standoff + " m";
  if (S.isSequence && S.seq) {
    $("frameVal").textContent = `${S.frameIndex + 1} / ${S.seq.header.nFrames}`;
  }
  const lib = S.traceLibs.get(
    (S.loaded?.header.dataset === "kitti" ? "kitti" : "nuscenes") + "-" + S.difficulty
  );
  if (lib) {
    $("traceIdx").max = String(lib.lib.header.nTraces - 1);
    const shown = S.traceIdx == null ? S.last?.result?.placed?.anno?.traceIdx : S.traceIdx;
    if (shown != null) $("traceIdx").value = String(shown);
    $("traceSpec").textContent =
      (S.traceIdx == null ? "random #" : "#") + (shown ?? 0) +
      (lib.lib.header.counts[shown ?? 0] != null ? ` · ${lib.lib.header.counts[shown ?? 0]} pts` : "");
  }
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
  try {
    S.seqManifest = await loadSequenceManifest("assets/sequences/manifest.json");
  } catch {
    S.seqManifest = [];
  }

  const gStatic = document.createElement("optgroup");
  gStatic.label = "Static frames";
  for (const m of S.manifest) {
    const o = document.createElement("option");
    o.value = m.slug;
    o.textContent = m.label;
    gStatic.appendChild(o);
  }
  $("scene").appendChild(gStatic);
  if (S.seqManifest.length) {
    const gSeq = document.createElement("optgroup");
    gSeq.label = "Driving sequences";
    for (const m of S.seqManifest) {
      const o = document.createElement("option");
      o.value = m.slug;
      o.textContent = `${m.label} · ${m.nFrames} frames`;
      gSeq.appendChild(o);
    }
    $("scene").appendChild(gSeq);
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

  seg(["atkInject", "atkRemove"], async (i) => {
    S.attack = i === 0 ? "inject" : "remove";
    $("injectOpts").style.display = i === 0 ? "" : "none";
    $("removeOpts").style.display = i === 0 ? "none" : "";
    await prepareSequenceAttack();
    await recompute();
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
  bind("azWidth", (v) => {
    // ATLAS accepts only the tabulated widths (argparse choices=sorted(PRR))
    S.azWidthDeg = ATLAS_AZ_WIDTHS.reduce(
      (b, w) => (Math.abs(w - v) < Math.abs(b - v) ? w : b), ATLAS_AZ_WIDTHS[0]
    );
    $("azWidth").value = String(S.azWidthDeg);
  });
  bind("pRemove", (v) => { S.pRemove = v; S.pRemoveOverride = true; });
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
  $("reroll").onclick = async () => {
    // back to ATLAS's own sampling for trace, offset and yaw
    S.seed = (S.seed * 1103515245 + 12345) & 0x7fffffff;
    S.traceIdx = null;
    S.phantomAt = null;
    buildSchedule();
    await prepareSequenceAttack();
    await recompute();
  };
  $("traceIdx").oninput = (e) => {
    S.traceIdx = parseInt(e.target.value, 10);
    labels();
    recompute();
  };
  const stepTrace = (d) => {
    const lib = S.traceLibs.get((S.loaded?.header.dataset === "kitti" ? "kitti" : "nuscenes") + "-" + S.difficulty);
    const n = lib ? lib.lib.header.nTraces : 200;
    const cur = S.traceIdx == null ? S.last?.result?.placed?.anno?.traceIdx ?? 0 : S.traceIdx;
    S.traceIdx = (cur + d + n) % n;
    labels();
    recompute();
  };
  $("frameIdx").oninput = async (e) => {
    await loadKeyframe(parseInt(e.target.value, 10));
    labels();
    await recompute();
  };
  seg(["placeEgo", "placeWorld"], async (i) => {
    S.placement = i === 0 ? "relative" : "global";
    await prepareSequenceAttack();
    await recompute();
  });
  $("schedMode").onchange = async (e) => {
    S.schedMode = e.target.value;
    buildSchedule();
    await prepareSequenceAttack();
    await recompute();
  };
  $("tracePrev").onclick = () => stepTrace(-1);
  $("traceNext").onclick = () => stepTrace(1);
  bind("standoff", (v) => { S.standoff = v; S.attackerWorld = null; });
  seg(["viewAtlas", "viewPhysical"], (i) => {
    S.gateByReach = i === 1;
    draw();
  });
  seg(["atkStatic", "atkDynamic"], (i) => {
    S.attackerMode = ATTACKER_MODES[i === 0 ? 0 : 1];
    S.attackerWorld = null; // re-anchor from the current aim
    recompute();
  });
  $("autoTarget").onclick = async () => {
    autoTarget(true);
    await prepareSequenceAttack();
    await recompute();
  };

  // click: hit a vehicle -> aim at it; otherwise move the spoofer
  const cv = $("cvClean");
  cv.onclick = (ev) => {
    const r = cv.getBoundingClientRect();
    const sx = ((ev.clientX - r.left) * cv.width) / r.width;
    const sy = ((ev.clientY - r.top) * cv.height) / r.height;
    const [wx, wy] = makeView(cv, S.viewRange).inv(sx, sy);

    // Shift-click plants the attacker. A roadside one keeps that WORLD position,
    // so it stays put while the ego drives on.
    if (ev.shiftKey) {
      S.attackerWorld = worldFromEgoPoint([wx, wy, 0], S.loaded.header.pose || null);
      recompute();
      return;
    }

    const boxes = (S.loaded.header.boxes || []).map(normalizeBox);
    if (S.attack === "inject") {
      // place the phantom where you clicked
      S.phantomAt = [wx, wy];
      prepareSequenceAttack().then(recompute);
      return;
    } else {
      const hit = pickBoxAt(wx, wy, boxes);
      if (hit >= 0) {
        S.targetIndex = hit;
        updateTargetInfo();
        prepareSequenceAttack().then(recompute);
        return;
      } else if (S.overlay) {
        S.spooferPos = [wx, wy, S.spooferPos[2]];
      }
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
  const known = (sl) =>
    S.manifest.some((m) => m.slug === sl) || S.seqManifest.some((m) => m.slug === sl);
  const initial = known(scene) ? scene : S.manifest[0].slug;
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
  if (q.has("schedule") && ["always", "continuous", "intermittent"].includes(q.get("schedule"))) {
    S.schedMode = q.get("schedule");
    $("schedMode").value = S.schedMode;
  }
  if (q.get("placement") === "global") {
    S.placement = "global";
    $("placeEgo").setAttribute("aria-pressed", "false");
    $("placeWorld").setAttribute("aria-pressed", "true");
  }
  if (q.has("frame")) S.pendingFrame = parseInt(q.get("frame"), 10);
  // Deterministic sweep position, for screenshots and headless verification:
  // ?sweptTo=90 marks the first 90 degrees as observed and stops there.
  if (q.has("sweptTo")) S.pendingSweptTo = (parseFloat(q.get("sweptTo")) * Math.PI) / 180;
  if (q.get("producible") === "1") {
    S.gateByReach = true;
    $("viewAtlas").setAttribute("aria-pressed", "false");
    $("viewPhysical").setAttribute("aria-pressed", "true");
  }
  if (q.get("attacker") === "dynamic") {
    S.attackerMode = "dynamic";
    $("atkStatic").setAttribute("aria-pressed", "false");
    $("atkDynamic").setAttribute("aria-pressed", "true");
  }
  if (q.has("speed")) {
    S.speed = Math.max(0.001, parseFloat(q.get("speed")));
    $("speed").value = String(Math.log10(S.speed));
  }
  if (q.get("spin") === "0") {
    S.spinning = false;
    $("fullBtn").click();
  }

  $("scene").value = initial;
  labels();
  await selectScene(initial);

  if (S.isSequence && S.pendingFrame != null) {
    await loadKeyframe(S.pendingFrame);
    $("frameIdx").value = String(S.frameIndex);
    await recompute();
  }

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
    await prepareSequenceAttack();
    await recompute();
  }

  if (S.pendingSweptTo != null && S.accum && S.last) {
    resetAccumulator(S.accum);
    markSwept(S.accum, 0, S.pendingSweptTo, S.layerId, S.last.sensor);
    S.sweepAz = S.pendingSweptTo;
    S.frozen = true; // hold this position so the view can be inspected
    draw();
  }

  labels();
  requestAnimationFrame(loop);
}
