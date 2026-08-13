/**
 * Sensor models.
 *
 * A spoofer's capability is bounded by the victim's scan lattice and pulse
 * timing, so the sensor spec is the foundation of every physical constraint
 * here. The hardware primitive is the *point rate* (points/sec, fixed by the
 * firing electronics); azimuth resolution then falls out of the spin rate:
 *
 *     columns/revolution = pointRate / (nBeams * hz)
 *
 * which is why a VLP-16 has 0.2 deg resolution at 10 Hz and 0.4 deg at 20 Hz.
 * For sensors whose lattice was measured from a real dataset we invert this and
 * take the measured azimuth resolution as given, deriving the point rate (which
 * then lands on the manufacturer's quoted figure - see sensor.test.js).
 */
import {
  HDL32E_INCLINATIONS,
  HDL64E_INCLINATIONS,
  HDL32E_AZ_RES,
  HDL64E_AZ_RES,
  uniformInclinations,
} from "./sensor-assets.js";
import { TWO_PI } from "./polar.js";

/** Speed of light in vacuum (m/s). Range = c * ToF / 2. */
export const C_LIGHT = 299792458;

const DEG = Math.PI / 180;

/**
 * Raw specifications. Exactly one of `azRes` (measured lattice) or `pointRate`
 * (datasheet firing rate) is the primitive; the other is derived.
 *
 * minRange is the Minimum Operational Threshold: returns closer than this are
 * discarded in firmware. Cao et al. (USENIX Sec '23) exploit precisely this to
 * turn saturating injection into *removal* on the VLP-16, where it is 40 cm.
 *
 * pTwo is the probability a second, farther echo survives alongside the first
 * on the same ray. Note this is NOT ATLAS's P_TWO value: the generator uses
 * 0.01 for Waymo (measuring how rarely Waymo reports a second return) and 0 for
 * single-return nuScenes/KITTI, whereas a sensor genuinely configured for dual
 * return reports ~2.03 returns/ray in the real capture, i.e. pTwo near 1.
 */
export const SENSORS = {
  vlp16: {
    name: "Velodyne VLP-16",
    dataset: null,
    nBeams: 16,
    inclinations: uniformInclinations(16, -15, 15),
    pointRate: 300000,
    hz: 10,
    maxRange: 100,
    minRange: 0.4,
    returnMode: "strongest",
    wavelengthNm: 903,
    receiverApertureM: 0.02,
    beamDivergenceMrad: 3.0,
    intensityScaleMax: 255,
    provenance: "nominal",
    note: "Sensor used in the Cao et al. physical removal-attack capture.",
  },
  hdl32e: {
    name: "Velodyne HDL-32E (nuScenes)",
    dataset: "nuscenes",
    nBeams: 32,
    inclinations: HDL32E_INCLINATIONS,
    azRes: HDL32E_AZ_RES,
    hz: 10,
    maxRange: 100,
    minRange: 0.5,
    returnMode: "strongest",
    wavelengthNm: 903,
    receiverApertureM: 0.022,
    beamDivergenceMrad: 3.0,
    intensityScaleMax: 255,
    provenance: "measured",
  },
  hdl64e: {
    name: "Velodyne HDL-64E (KITTI)",
    dataset: "kitti",
    nBeams: 64,
    inclinations: HDL64E_INCLINATIONS,
    azRes: HDL64E_AZ_RES,
    hz: 10,
    maxRange: 120,
    minRange: 0.9,
    returnMode: "strongest",
    wavelengthNm: 905,
    receiverApertureM: 0.025,
    beamDivergenceMrad: 2.0,
    intensityScaleMax: 1.0,
    provenance: "measured",
  },
  os1_64: {
    name: "Ouster OS1-64",
    dataset: null,
    nBeams: 64,
    inclinations: uniformInclinations(64, -22.5, 22.5),
    azRes: TWO_PI / 1024,
    hz: 10,
    maxRange: 120,
    minRange: 0.3,
    returnMode: "strongest",
    wavelengthNm: 865,
    receiverApertureM: 0.01,
    beamDivergenceMrad: 1.5,
    intensityScaleMax: 255,
    provenance: "nominal",
    note: "Fixed 1024-column lattice; resolution does not track spin rate.",
  },
  vlp32c: {
    name: "Velodyne VLP-32C",
    dataset: null,
    nBeams: 32,
    inclinations: uniformInclinations(32, -25, 15),
    pointRate: 600000,
    hz: 10,
    maxRange: 200,
    minRange: 0.4,
    returnMode: "strongest",
    wavelengthNm: 903,
    receiverApertureM: 0.022,
    beamDivergenceMrad: 3.0,
    intensityScaleMax: 255,
    provenance: "nominal",
  },
};

export function listSensors() {
  return Object.keys(SENSORS);
}

/**
 * Build a fully derived sensor spec, optionally overriding fields.
 * Never mutates the registry.
 *
 * Overrides: hz, returnMode, maxRange, minRange, azResDeg, pointRate,
 * inclinations.
 */
export function configureSensor(id, overrides = {}) {
  const base = SENSORS[id];
  if (!base) {
    throw new Error(
      'unknown sensor id "' + id + '" (known: ' + listSensors().join(", ") + ")"
    );
  }
  const s = { id, ...base, ...overrides };

  // --- azimuth lattice ---------------------------------------------------
  // An explicit azimuth resolution wins and forces the point rate to be
  // rederived; otherwise a datasheet point rate defines the lattice.
  let azRes;
  let pointRate;
  const azOverridden = overrides.azResDeg != null || overrides.azRes != null;
  if (overrides.azResDeg != null) {
    azRes = overrides.azResDeg * DEG;
  } else if (overrides.azRes != null) {
    azRes = overrides.azRes;
  } else if (base.azRes != null) {
    azRes = base.azRes;
  } else {
    const cols = (overrides.pointRate ?? base.pointRate) / (s.nBeams * s.hz);
    azRes = TWO_PI / cols;
  }
  const azCellCount = Math.ceil(TWO_PI / azRes);

  if (azOverridden || base.azRes != null) {
    pointRate = s.nBeams * azCellCount * s.hz;
  } else {
    pointRate = overrides.pointRate ?? base.pointRate;
  }

  // --- pulse timing ------------------------------------------------------
  const revolutionPeriod = 1 / s.hz;
  const laserPRF = azCellCount * s.hz;
  const firingPeriod = 1 / laserPRF;
  const maxToF = (2 * s.maxRange) / C_LIGHT;
  // The receiver only listens for a window after each firing. Give it 50%
  // headroom over its own max range, but it can never outlast the next pulse.
  const rangeGate = Math.min(1.5 * maxToF, firingPeriod);

  const pTwo = s.returnMode === "dual" ? 0.98 : 0;

  return Object.freeze({
    ...s,
    azRes,
    azCellCount,
    pointRate,
    pointsPerRevolution: s.nBeams * azCellCount,
    revolutionPeriod,
    laserPRF,
    firingPeriod,
    rangeGate,
    maxToF,
    pTwo,
    vfovDeg: (s.inclinations[s.inclinations.length - 1] - s.inclinations[0]) / DEG,

    /** Time within a revolution at which azimuth column `i` is sampled. */
    timeOfAzCell(i) {
      return (i / azCellCount) * revolutionPeriod;
    },
    rangeToToF(r) {
      return (2 * r) / C_LIGHT;
    },
    tofToRange(t) {
      return (t * C_LIGHT) / 2;
    },
  });
}

const cache = new Map();

/** Derived spec for a registry sensor, at its default configuration. */
export function getSensor(id) {
  if (!cache.has(id)) cache.set(id, configureSensor(id));
  return cache.get(id);
}
