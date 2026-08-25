/**
 * Phantom intensity models.
 *
 * ATLAS writes phantom intensity sampled from range-conditioned statistics of
 * real returns, which keeps an injected object in-distribution so that measured
 * attack success reflects geometry rather than an obvious brightness tell.
 *
 * The real spoofer does the opposite: its returns saturate (median 255/255,
 * fraction at full scale = 1.000). ATLAS_REAL_WORLD_VALIDATION.md shows that
 * forcing saturated intensity collapses attack success by 85-95% across pillar,
 * sparse-voxel and transformer detectors - and that the response is *flat* over
 * I = 15..100 with a cliff past ~100, which is exactly the p99 of real nuScenes
 * car returns. So the effect is out-of-distribution detectability in the top 1%
 * tail, not "brighter is easier to spot".
 *
 * All three modes are offered because the choice changes what the sandbox is
 * measuring: `range-conditioned` for a geometry benchmark, `saturated` to model
 * the physically real spoofer, `constant` to reproduce the dose-response sweep.
 */
import { INTENSITY_BANDS } from "./sensor-assets.js";

export const INTENSITY_MODES = ["range-conditioned", "saturated", "constant"];

/** p99 of real *car* returns, the threshold the ASR cliff sits at. */
const REAL_CAR_P99 = { nuscenes: 100, kitti: 0.5 };

export function realCarIntensityP99(dataset) {
  return REAL_CAR_P99[dataset] ?? 100;
}

export function datasetScaleMax(dataset) {
  return INTENSITY_BANDS[dataset]?.scaleMax ?? 255;
}

/**
 * Band containing `r`, half-open [rMin, rMax) as in ATLAS's
 * get_stats_for_range. Clamps to the outermost band rather than returning null,
 * so a distant phantom still gets plausible intensity instead of silently
 * keeping whatever the source trace had.
 */
export function bandForRange(dataset, r) {
  const spec = INTENSITY_BANDS[dataset];
  if (!spec) throw new Error('no intensity statistics for dataset "' + dataset + '"');
  const bands = spec.bands;
  for (const b of bands) {
    if (r >= b.rMin && r < b.rMax) return b;
  }
  return r < bands[0].rMin ? bands[0] : bands[bands.length - 1];
}

/** Standard normal via Box-Muller, driven by an injected uniform generator. */
function gaussian(next) {
  let u = next();
  if (u <= 0) u = Number.MIN_VALUE;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
}

/** One lognormal draw, clipped to the band's measured percentiles. */
export function sampleIntensity({ dataset, r, next }) {
  const b = bandForRange(dataset, r);
  const v = Math.exp(b.logMu + b.logSigma * gaussian(next));
  return Math.min(b.p99, Math.max(b.p1, v));
}

/**
 * ATLAS's range-conditioned intensity, faithfully.
 *
 * Both drivers compute ONE mean range over the whole phantom
 * (`r_avg = avg_spoof_range(spf)`), look up the single band containing it, and
 * draw every point from that band. Sampling per point instead would mix bands
 * across a phantom straddling a boundary.
 *
 * If the mean range falls past the last band, `get_stats_for_range` returns None
 * and the caller simply does not overwrite the intensity column - so the trace's
 * own recorded intensity survives. `fallback` is that original intensity.
 */
export function atlasRangeConditionedIntensity({ dataset, meanRange, count, next, fallback }) {
  const spec = INTENSITY_BANDS[dataset];
  if (!spec) throw new Error('no intensity statistics for dataset "' + dataset + '"');
  const bands = spec.bands;
  const last = bands[bands.length - 1];
  if (meanRange >= last.rMax || meanRange < bands[0].rMin) {
    // outside the tabulated range: leave it alone, as ATLAS does
    return fallback ? Float32Array.from(fallback.slice(0, count)) : new Float32Array(count);
  }
  const b = bandForRange(dataset, meanRange);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const v = Math.exp(b.logMu + b.logSigma * gaussian(next));
    out[i] = Math.min(b.p99, Math.max(b.p1, v));
  }
  return out;
}

/**
 * Intensity for every phantom point.
 * @returns {Float32Array}
 */
export function assignIntensity({ mode, dataset, ranges, next, constantValue = 0, scaleMax = null }) {
  if (!INTENSITY_MODES.includes(mode)) {
    throw new Error(
      'unknown intensity mode "' + mode + '" (known: ' + INTENSITY_MODES.join(", ") + ")"
    );
  }
  const n = ranges.length;
  const out = new Float32Array(n);

  if (mode === "saturated") {
    const full = scaleMax ?? datasetScaleMax(dataset);
    out.fill(full);
    return out;
  }
  if (mode === "constant") {
    out.fill(constantValue);
    return out;
  }
  for (let i = 0; i < n; i++) {
    out[i] = sampleIntensity({ dataset, r: ranges[i], next });
  }
  return out;
}
