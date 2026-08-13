/**
 * GENERATED FILE - do not edit by hand.
 * Regenerate with:  python3 tools/gen_sensor_assets.py
 *
 * Beam inclinations, azimuth resolution and range-conditioned intensity
 * statistics measured from the real datasets, emitted from the same .npy assets
 * that atlas_generation/*_atlas_common.py loads. Sources:
 *   atlas_generation/nuscenes_beam_inclinations.npy  sha256:2460e5d5
 *   atlas_generation/nuscenes_azimuth_res.npy  sha256:e9fa4eff
 *   atlas_generation/nuscenes_range_conditioned_stats.npy  sha256:dde98850
 *   atlas_generation/assets_kitti/kitti_beam_inclinations.npy  sha256:37899d3a
 *   atlas_generation/assets_kitti/kitti_azimuth_res.npy  sha256:56f4f685
 *   atlas_generation/assets_kitti/kitti_range_conditioned_stats.npy  sha256:aab014ec
 *
 * `provenance: "measured"` means the inclinations were fitted from real dataset
 * returns. `"nominal"` means they are datasheet values expanded uniformly across
 * the vertical FOV - honest to the spec sheet, but not measured hardware.
 */

/** nuScenes LiDAR: Velodyne HDL-32E, 32 beams, measured inclinations (rad). */
export const HDL32E_INCLINATIONS = new Float64Array([
    -0.533554580, -0.512558985, -0.491062344, -0.468244916, -0.443615896, -0.423387049,
    -0.399204267, -0.375934054, -0.354002256, -0.330266342, -0.306405411, -0.282853889,
    -0.259226374, -0.235424532, -0.211805384, -0.188245352, -0.164526992, -0.141038819,
    -0.117407362, -0.094008836, -0.070609780, -0.047227561, -0.023704694, -0.000390098,
    0.022893384, 0.046370228, 0.069663512, 0.092958242, 0.116439969, 0.139756050,
    0.163067753, 0.186566521
]);

/** KITTI LiDAR: Velodyne HDL-64E, 64 beams, measured inclinations (rad). */
export const HDL64E_INCLINATIONS = new Float64Array([
    -0.418660551, -0.393842593, -0.380018026, -0.370360568, -0.358113199, -0.346367091,
    -0.336759943, -0.327756065, -0.320050120, -0.312084630, -0.303799555, -0.295037806,
    -0.285660669, -0.275862327, -0.267472198, -0.260125101, -0.252170227, -0.243665785,
    -0.234457098, -0.225579493, -0.217590310, -0.210423327, -0.204036965, -0.198261540,
    -0.191460621, -0.182769686, -0.173987255, -0.166812520, -0.159699325, -0.151620545,
    -0.142754905, -0.134766653, -0.129016817, -0.123839742, -0.118085703, -0.112699755,
    -0.107796122, -0.102736659, -0.097420807, -0.092021935, -0.086667020, -0.081326408,
    -0.076101718, -0.070897247, -0.065683656, -0.060537768, -0.055220074, -0.049869327,
    -0.044534448, -0.039073070, -0.033565889, -0.027800148, -0.021831502, -0.016073241,
    -0.010113871, -0.003759842, 0.002668807, 0.008893250, 0.015128963, 0.021637717,
    0.028082281, 0.034374690, 0.041066700, 0.108055745
]);

/** Measured azimuth resolution (radians per cell). */
export const HDL32E_AZ_RES = 0.0057873726;
export const HDL64E_AZ_RES = 0.0030159289;

/**
 * Range-conditioned intensity statistics, per dataset. A phantom's intensity is
 * drawn lognormal(logMu, logSigma) clipped to [p1, p99] for the band containing
 * its mean range - this is what keeps an injected object in-distribution.
 *
 * Note the differing scales: nuScenes intensity is 0-255, KITTI is normalised
 * 0-1. ATLAS_REAL_WORLD_VALIDATION.md measures detector ASR collapsing past
 * I ~= 100 on nuScenes, which is exactly the p99 of real car returns.
 */
export const INTENSITY_BANDS = {
  nuscenes: {
    scaleMax: 255,
    bands: [
      { rMin: 0, rMax: 10, logMu: 2.350447416, logSigma: 1.179853605, p1: 1, p99: 104, n: 6009059 },
      { rMin: 10, rMax: 20, logMu: 2.374062300, logSigma: 1.109894441, p1: 1, p99: 100, n: 1310776 },
      { rMin: 20, rMax: 30, logMu: 2.351169348, logSigma: 0.914421843, p1: 1, p99: 52, n: 615732 },
      { rMin: 30, rMax: 40, logMu: 2.790868282, logSigma: 0.647936867, p1: 4, p99: 61, n: 278926 },
      { rMin: 40, rMax: 50, logMu: 3.068240643, logSigma: 0.522349285, p1: 7, p99: 74, n: 135627 },
      { rMin: 50, rMax: 80, logMu: 3.381546259, logSigma: 0.412253814, p1: 12, p99: 85, n: 141379 }
    ],
  },
  kitti: {
    scaleMax: 1.0,
    bands: [
      { rMin: 0, rMax: 10, logMu: -1.371501803, logSigma: 0.538262235, p1: 0.04, p99: 0.58, n: 22929853 },
      { rMin: 10, rMax: 20, logMu: -1.211992741, logSigma: 0.455457138, p1: 0.05, p99: 0.68, n: 12397194 },
      { rMin: 20, rMax: 30, logMu: -1.302588463, logSigma: 0.428933369, p1: 0.08, p99: 0.64, n: 4244985 },
      { rMin: 30, rMax: 40, logMu: -1.533247828, logSigma: 0.479191767, p1: 0.06, p99: 0.58, n: 1378084 },
      { rMin: 40, rMax: 50, logMu: -1.736348271, logSigma: 0.524228560, p1: 0.05, p99: 0.7, n: 522163 },
      { rMin: 50, rMax: 80, logMu: -1.899957418, logSigma: 0.654276954, p1: 0.03, p99: 0.91, n: 234132 }
    ],
  },
};

/** Uniformly spaced inclinations across a vertical FOV, ascending (radians). */
export function uniformInclinations(nBeams, vfovMinDeg, vfovMaxDeg) {
  const out = new Float64Array(nBeams);
  const lo = (vfovMinDeg * Math.PI) / 180;
  const hi = (vfovMaxDeg * Math.PI) / 180;
  const step = nBeams > 1 ? (hi - lo) / (nBeams - 1) : 0;
  for (let i = 0; i < nBeams; i++) out[i] = lo + i * step;
  return out;
}
