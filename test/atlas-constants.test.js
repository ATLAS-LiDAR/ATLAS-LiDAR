import { describe, it, expect } from "./harness.js";
import { rng } from "./util.js";
import {
  PRR,
  pRemoveForWidth,
  ATLAS_AZ_WIDTHS,
  selectNearestFrontCar,
  NUSCENES_TRACK,
  WAYMO_TRACK,
} from "../src/core/atlas-removal.js";
import {
  RELATIVE_OFFSET,
  RELATIVE_OFFSET_BY_DATASET,
  placeTraceRelative,
} from "../src/core/atlas-inject.js";
import { atlasRangeConditionedIntensity } from "../src/core/intensity.js";

/**
 * These pin values that exist in the ATLAS source. They are transcription, not
 * judgement: if one of these fails, either the port drifted or the pipeline
 * changed, and both are worth knowing. Source locations are named so the claim is
 * checkable.
 */
describe("p_remove is derived from az_width, not chosen", () => {
  it("matches the PRR table verbatim", () => {
    // generate_data_removal_nuscenes.py:34 and generate_data_kitti.py:46
    expect(PRR).toEqual({ 10: 0.97, 20: 1.0, 30: 0.99, 40: 0.98, 50: 0.9, 60: 0.9 });
  });

  it("uses 1.00 at the default 20 degree width, not 0.96", () => {
    // 0.96 is removal.py's function default, which both drivers override.
    expect(pRemoveForWidth(20)).toBeCloseTo(1.0, 1e-12);
  });

  it("offers exactly the widths ATLAS allows", () => {
    // argparse: choices=sorted(PRR)
    expect(ATLAS_AZ_WIDTHS).toEqual([10, 20, 30, 40, 50, 60]);
  });

  it("shares one table between nuScenes and KITTI", () => {
    for (const w of ATLAS_AZ_WIDTHS) expect(pRemoveForWidth(w)).toBeCloseTo(PRR[w], 1e-12);
  });

  it("snaps an off-table width to the nearest tabulated one", () => {
    expect(pRemoveForWidth(22)).toBeCloseTo(PRR[20], 1e-12);
    expect(pRemoveForWidth(55)).toBeCloseTo(PRR[50], 1e-12);
    expect(pRemoveForWidth(3)).toBeCloseTo(PRR[10], 1e-12);
    expect(pRemoveForWidth(200)).toBeCloseTo(PRR[60], 1e-12);
  });

  it("is not monotonic in width, as measured", () => {
    // az20 is the most complete removal; the wide wedges drop to 0.90
    expect(pRemoveForWidth(20)).toBeGreaterThan(pRemoveForWidth(10));
    expect(pRemoveForWidth(60)).toBeLessThan(pRemoveForWidth(40));
  });
});

describe("injection placement differs by dataset", () => {
  it("uses the nuScenes window for nuScenes", () => {
    // nuscenes_atlas_common.place_trace_relative
    const w = RELATIVE_OFFSET_BY_DATASET.nuscenes;
    expect([w.xMin, w.xMax, w.yAbs, w.yawAbs]).toEqual([15, 20, 3, 0.2]);
  });

  it("uses the WIDER KITTI window, set by its camera FOV", () => {
    // kitti_atlas_common.place_trace: uniform(12,30), uniform(-6,6), yaw +-0.3
    const w = RELATIVE_OFFSET_BY_DATASET.kitti;
    expect([w.xMin, w.xMax, w.yAbs, w.yawAbs]).toEqual([12, 30, 6, 0.3]);
  });

  it("keeps the nuScenes window as the default for back-compatibility", () => {
    expect(RELATIVE_OFFSET.xMin).toBe(15);
  });

  it("places a KITTI phantom inside KITTI's window", () => {
    const traces = [{
      x: Float64Array.from([1, 0]), y: Float64Array.from([0, 1]),
      z: Float64Array.from([-1.2, -1.0]), intensity: Float32Array.from([1, 1]),
      box: [0, 0, -1.2, 4.2, 1.8, 1.6, 0],
    }];
    for (let seed = 1; seed <= 40; seed++) {
      const { box, anno } = placeTraceRelative(traces, { seed, dataset: "kitti" });
      expect(box[0]).toBeGreaterThanOrEqual(12);
      expect(box[0]).toBeLessThanOrEqual(30);
      expect(Math.abs(box[1])).toBeLessThanOrEqual(6);
      expect(Math.abs(box[6])).toBeLessThanOrEqual(0.3);
      expect(anno.dataset).toBe("kitti");
    }
  });

  it("places a nuScenes phantom in the narrower window", () => {
    const traces = [{
      x: Float64Array.from([1]), y: Float64Array.from([0]),
      z: Float64Array.from([-1.2]), intensity: Float32Array.from([1]),
      box: [0, 0, -1.2, 4.2, 1.8, 1.6, 0],
    }];
    for (let seed = 1; seed <= 40; seed++) {
      const { box } = placeTraceRelative(traces, { seed, dataset: "nuscenes" });
      expect(box[0]).toBeGreaterThanOrEqual(15);
      expect(box[0]).toBeLessThanOrEqual(20);
      expect(Math.abs(box[1])).toBeLessThanOrEqual(3);
    }
  });
});

describe("phantom intensity: ONE band from the mean range", () => {
  // Both drivers compute r_avg over the whole phantom and draw every point from
  // that single band. Sampling per point would mix bands across a phantom that
  // straddles a boundary.
  it("draws every point from the band containing the mean range", () => {
    const next = rng(4);
    const out = atlasRangeConditionedIntensity({
      dataset: "nuscenes", meanRange: 15, count: 500, next,
      fallback: new Float32Array(500).fill(7),
    });
    expect(out).toHaveLength(500);
    // band (10,20) for nuScenes: clipped to [p1, p99]
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).toBeGreaterThanOrEqual(1);
      expect(out[i]).toBeLessThanOrEqual(120);
    }
  });

  it("leaves intensity UNTOUCHED past the last band, as ATLAS does", () => {
    // get_stats_for_range returns None beyond 80 m, and the caller then simply
    // does not overwrite column 3 - so the trace's own recorded intensity stays.
    const original = Float32Array.from([11, 22, 33]);
    const out = atlasRangeConditionedIntensity({
      dataset: "nuscenes", meanRange: 500, count: 3, next: rng(1), fallback: original,
    });
    expect(Array.from(out)).toEqual([11, 22, 33]);
  });

  it("uses the dataset's own statistics", () => {
    const k = atlasRangeConditionedIntensity({
      dataset: "kitti", meanRange: 15, count: 200, next: rng(2),
      fallback: new Float32Array(200),
    });
    // KITTI intensity is normalised 0-1, nuScenes is 0-255
    for (let i = 0; i < k.length; i++) expect(k[i]).toBeLessThanOrEqual(1);
  });
});

describe("removal target selection is dataset-specific", () => {
  it("KITTI picks the NEAREST front car beyond 10 m, by 2D range", () => {
    // generate_data_kitti.py: cars[:,0] > 10 then argmin of |xy|
    const boxes = [
      [8, 0, 0, 4, 2, 1.6, 0, 1],    // too close
      [40, 0, 0, 4, 2, 1.6, 0, 1],   // far
      [15, 3, 0, 4, 2, 1.6, 0, 1],   // nearest qualifying
      [-30, 0, 0, 4, 2, 1.6, 0, 1],  // behind
    ];
    const r = selectNearestFrontCar(boxes, { minX: 10 });
    expect(r.index).toBe(2);
  });

  it("KITTI ignores range in z, measuring only the footprint distance", () => {
    const boxes = [
      [20, 0, 50, 4, 2, 1.6, 0, 1], // absurd height, but 20 m away in xy
      [25, 0, 0, 4, 2, 1.6, 0, 1],
    ];
    expect(selectNearestFrontCar(boxes, { minX: 10 }).index).toBe(0);
  });

  it("KITTI returns nothing when no car is in front", () => {
    expect(selectNearestFrontCar([[5, 0, 0, 4, 2, 1.6, 0, 1]], { minX: 10 }).index).toBe(null);
    expect(selectNearestFrontCar([], { minX: 10 }).index).toBe(null);
  });

  it("pins the nuScenes sequence tracking constants", () => {
    // generate_data_removal_nuscenes.py: desired_range=40, min_range=10,
    // negative=True, NUSC_KEYFRAME_DT=0.5, NUSC_MAX_STEP_M=12.0
    expect(NUSCENES_TRACK.desiredRange).toBe(40);
    expect(NUSCENES_TRACK.minRange).toBe(10);
    expect(NUSCENES_TRACK.negative).toBe(true);
    expect(NUSCENES_TRACK.dt).toBeCloseTo(0.5, 1e-12);
    expect(NUSCENES_TRACK.maxStep).toBeCloseTo(12.0, 1e-12);
  });

  it("keeps the Waymo constants distinct, since they differ", () => {
    // generate_data_removal.py: desired_range=60, max_step=6, dt default 0.1
    expect(WAYMO_TRACK.desiredRange).toBe(60);
    expect(WAYMO_TRACK.maxStep).toBeCloseTo(6.0, 1e-12);
    expect(WAYMO_TRACK.dt).toBeCloseTo(0.1, 1e-12);
    expect(NUSCENES_TRACK.maxStep).not.toBe(WAYMO_TRACK.maxStep);
  });
});
