#!/usr/bin/env python
"""
Build per-class phantom trace libraries for the nuScenes ATLAS injection
generators, for classes other than 'car' (pedestrian, cyclist, ...).

build_nuscenes_assets.py only extracts one GT class and writes the beam/intensity
assets alongside it. This script reuses its extraction convention but:

  * writes trace_<name>_{easy,medium,hard}.pkl (the sensor assets are untouched),
  * filters by nuScenes attribute, so 'cyclist' = bicycle WITH a rider
    (nuScenes 'bicycle' boxes also cover parked, riderless bicycles),
  * bins easy/medium/hard by point-density quantiles WITHIN each class, so a
    pedestrian 'easy' is dense for a pedestrian rather than for a car.

The trace dicts keep the exact format generate_data_{global,relative}_nuscenes.py
reads ({"points", "box", "n_points", "range"}), plus "cls", so they can be passed
straight to --trace_file.

Example (attacker library from train only, to avoid leakage into val):
  python build_nuscenes_traces_multiclass.py \
      --dataroot ../data/nuscenes/v1.0-trainval \
      --classes pedestrian cyclist --out_dir assets_trainval
"""
import os
import json
import argparse
import pickle as pkl

import numpy as np

from build_nuscenes_assets import load_raw

# name -> (gt_names allowed, attribute names allowed or None for any)
CLASS_SPECS = {
    "pedestrian": (["pedestrian"], ["pedestrian.standing", "pedestrian.moving"]),
    "cyclist": (["bicycle"], ["cycle.with_rider"]),
    "motorcyclist": (["motorcycle"], ["cycle.with_rider"]),
    "car": (["car"], None),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True,
                   help="nuScenes v1.0-trainval dir with samples/ and the info pkls")
    p.add_argument("--info", default="nuscenes_infos_10sweeps_train.pkl",
                   help="info pkl to extract traces from (train = no val leakage)")
    p.add_argument("--meta_dir", default=None,
                   help="dir holding sample_annotation.json / attribute.json "
                        "(default: <dataroot>/v1.0-trainval)")
    p.add_argument("--classes", nargs="+", default=["pedestrian", "cyclist"],
                   choices=sorted(CLASS_SPECS))
    p.add_argument("--min_points", type=int, default=5)
    p.add_argument("--max_per_bucket", type=int, default=4000)
    p.add_argument("--out_dir", default=".")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_ann_attributes(meta_dir):
    """sample_annotation token -> set of attribute names."""
    with open(os.path.join(meta_dir, "attribute.json")) as f:
        tok2name = {a["token"]: a["name"] for a in json.load(f)}
    with open(os.path.join(meta_dir, "sample_annotation.json")) as f:
        anns = json.load(f)
    return {a["token"]: {tok2name[t] for t in a["attribute_tokens"]} for a in anns}


def points_in_boxes(pts, boxes7):
    """(M, N) bool, same in-box test as pcdet's points_in_boxes_cpu.

    pcdet's CPU op takes ~2.5 s on one nuScenes scan (~20 h over train); this
    crops to each box's bounding circle first and is effectively free.
    """
    margin = 1e-2  # pcdet's MARGIN on x/y
    out = np.zeros((boxes7.shape[0], pts.shape[0]), dtype=bool)
    for bi, (cx, cy, cz, dx, dy, dz, rz) in enumerate(boxes7):
        reach = 0.5 * np.hypot(dx, dy) + margin
        near = np.nonzero((np.abs(pts[:, 0] - cx) <= reach) &
                          (np.abs(pts[:, 1] - cy) <= reach) &
                          (np.abs(pts[:, 2] - cz) <= dz / 2.0))[0]
        sx, sy = pts[near, 0] - cx, pts[near, 1] - cy
        c, s = np.cos(-rz), np.sin(-rz)
        lx, ly = sx * c - sy * s, sx * s + sy * c
        out[bi, near] = (np.abs(lx) < dx / 2.0 + margin) & (np.abs(ly) < dy / 2.0 + margin)
    return out


def extract_traces(dataroot, infos, ann_attrs, specs, min_points):
    traces = {name: [] for name in specs}
    for i, info in enumerate(infos):
        gt_names = info["gt_names"]
        tokens = info["gt_boxes_token"]
        sel = {}
        for name, (names, attrs) in specs.items():
            m = np.isin(gt_names, names)
            if attrs is not None:
                m &= np.array([bool(ann_attrs.get(t, set()) & set(attrs))
                               for t in tokens], dtype=bool)
            if m.any():
                sel[name] = m
        if not sel:
            continue

        raw = load_raw(dataroot, info["lidar_path"])
        pts = raw[:, :4].astype(np.float32)  # x,y,z,intensity (drop ring)
        for name, m in sel.items():
            boxes7 = info["gt_boxes"][m][:, :7].astype(np.float32)
            idx = points_in_boxes(pts, boxes7)
            for bi in range(boxes7.shape[0]):
                in_box = idx[bi]
                n = int(in_box.sum())
                if n < min_points:
                    continue
                # [x,y,z,intensity, col4]; col4 = 0.0 (timestamp placeholder)
                obj_pts5 = np.hstack([pts[in_box], np.zeros((n, 1), np.float32)])
                box = boxes7[bi].copy()
                traces[name].append({"points": obj_pts5, "box": box, "n_points": n,
                                     "range": float(np.hypot(box[0], box[1])),
                                     "cls": name})
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(infos)} infos | " +
                  " ".join(f"{k}={len(v)}" for k, v in traces.items()))
    return traces


def write_buckets(name, traces, out_dir, seed, max_per_bucket):
    n_pts = np.array([t["n_points"] for t in traces])
    print(f"[traces] {name}: {len(traces)} objects | points/obj min/med/max = "
          f"{n_pts.min()}/{int(np.median(n_pts))}/{n_pts.max()}")
    q33, q66 = np.percentile(n_pts, [33, 66])
    buckets = {
        "easy":   [t for t in traces if t["n_points"] >= q66],
        "medium": [t for t in traces if q33 <= t["n_points"] < q66],
        "hard":   [t for t in traces if t["n_points"] < q33],
    }
    rng = np.random.default_rng(seed)
    for diff, lst in buckets.items():
        if len(lst) > max_per_bucket:
            sel = rng.choice(len(lst), size=max_per_bucket, replace=False)
            lst = [lst[i] for i in sel]
        path = os.path.join(out_dir, f"trace_{name}_{diff}.pkl")
        with open(path, "wb") as f:
            pkl.dump(lst, f)
        rr = np.array([t["range"] for t in lst]) if lst else np.array([0.0])
        nn = np.array([t["n_points"] for t in lst]) if lst else np.array([0])
        print(f"[traces] {name}_{diff:6s}: {len(lst):4d} objs | "
              f"pts med {int(np.median(nn))} | mean range {rr.mean():.1f}m -> {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    meta_dir = args.meta_dir or os.path.join(args.dataroot, "v1.0-trainval")

    with open(os.path.join(args.dataroot, args.info), "rb") as f:
        infos = pkl.load(f)
    print(f"traces from {len(infos)} infos ({args.info}) | classes {args.classes}")

    ann_attrs = load_ann_attributes(meta_dir)
    specs = {c: CLASS_SPECS[c] for c in args.classes}
    traces = extract_traces(args.dataroot, infos, ann_attrs, specs, args.min_points)
    for name in args.classes:
        if not traces[name]:
            print(f"[traces] {name}: no objects found, skipped")
            continue
        write_buckets(name, traces[name], args.out_dir, args.seed, args.max_per_bucket)
    print("done.")


if __name__ == "__main__":
    main()
