"""Build a VAL-split object bank (GT-database style) and ATLAS trace banks.

Mirrors how traces_easy/medium/hard.pkl were made (generate_data_relative.ipynb,
commented trace-collection cell): iterate single-frame val samples
(dataset_sf[i], the same cfg the generator uses for scene points), and for each
labelled box take every point inside it (no enlargement). Differences from the
original bank:
  * source = val segments EXCLUDING the attacked ATLAS segments (the original
    banks came from val frames 0-198, i.e. one non-attacked segment), so pasted
    objects never come from the scene they are pasted into;
  * every --frame_stride-th frame of every source segment (more distinct
    objects, fewer near-duplicate copies of the same tracked object);
  * Vehicle AND Cyclist; records carry segment / frame / obj_id / range.

Outputs (in --out_dir):
  object_bank_val.pkl          list of {class, segment, frame_id, frame, obj_id,
                               box (8: x,y,z,dx,dy,dz,yaw,label), points (N,5),
                               num_points, range}     -> --object_bank for gt_paste
  traces_val_<cls>_<name>.pkl  ATLAS-format trace banks (same keys as
                               traces_*.pkl: frame, box, points, num_points, +
                               segment, obj_id), one per --trace_band
  bank_summary.json            counts, distinct objects, point/range stats
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LABELS = {"Vehicle": 1, "Pedestrian": 2, "Cyclist": 3}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--openpcdet", required=True)
    p.add_argument("--cfg", required=True, help="single-frame cfg (generator's --cp_sf_cfg)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--exclude_dataset_dir", required=True,
                   help="ATLAS run dataset dir; its segments are excluded as sources")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--classes", nargs="+", default=["Vehicle", "Cyclist"])
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--max_range", type=float, default=60.0)
    p.add_argument("--min_points", type=int, default=5)
    p.add_argument("--trace_band", action="append", default=[],
                   help="CLASS:NAME:MINPTS:MAXPTS[:COUNT]  e.g. Cyclist:hard:100:300:300")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def points_in_box(points, box):
    """Exact copy of the original trace-collection test."""
    cx, cy, cz, dx, dy, dz, heading = box[:7]
    shifted = points[:, :3] - np.array([cx, cy, cz])
    c, s = np.cos(-heading), np.sin(-heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = shifted @ R.T
    return ((np.abs(local[:, 0]) <= dx / 2) & (np.abs(local[:, 1]) <= dy / 2)
            & (np.abs(local[:, 2]) <= dz / 2))


def main():
    a = parse_args()
    sys.path.insert(0, a.openpcdet)
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils

    cfg_from_yaml_file(a.cfg, cfg)
    cfg.DATA_CONFIG.DATA_PATH = a.data_path
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                     batch_size=1, dist=False, workers=0, logger=logger,
                                     training=False)
    class_names = list(cfg.CLASS_NAMES)
    wanted = {class_names.index(c) + 1: c for c in a.classes}
    excluded = {f[:-4] for f in os.listdir(a.exclude_dataset_dir) if f.endswith(".pkl")}

    idxs = []
    for i, info in enumerate(dataset.infos):
        seg = str(info["point_cloud"]["lidar_sequence"])
        if seg in excluded:
            continue
        if int(info["point_cloud"]["sample_idx"]) % a.frame_stride == 0:
            idxs.append(i)
    src_segs = {str(dataset.infos[i]["point_cloud"]["lidar_sequence"]) for i in idxs}
    logger.info("Sources: %d frames from %d val segments (excluded %d attacked segments)",
                len(idxs), len(src_segs), len(excluded))
    if src_segs & excluded:
        raise RuntimeError("attacked segment leaked into source set")

    bank = []
    for n, i in enumerate(idxs):
        info = dataset.infos[i]
        d = dataset[i]
        pts = np.asarray(d["points"], dtype=np.float32)
        boxes = np.asarray(d.get("gt_boxes", np.zeros((0, 8))), dtype=np.float32)
        an = info["annos"]
        ann_xyz = np.asarray(an["gt_boxes_lidar"], dtype=np.float32)[:, :3]
        for box in boxes:
            label = int(box[-1])
            if label not in wanted:
                continue
            rng_m = float(np.hypot(box[0], box[1]))
            if rng_m > a.max_range:
                continue
            m = points_in_box(pts, box)
            k = int(m.sum())
            if k < a.min_points:
                continue
            j = int(np.argmin(np.linalg.norm(ann_xyz - box[:3], axis=1))) if len(ann_xyz) else -1
            obj_id = str(an["obj_ids"][j]) if j >= 0 and "obj_ids" in an else None
            bank.append({
                "class": wanted[label],
                "segment": str(info["point_cloud"]["lidar_sequence"]),
                "frame_id": str(info["frame_id"]),
                "frame": int(info["point_cloud"]["sample_idx"]),
                "obj_id": obj_id,
                "box": np.concatenate([box[:7], [label]]).astype(np.float32),
                "points": pts[m].copy(),
                "num_points": k,
                "range": rng_m,
            })
        if n % 500 == 0:
            logger.info("%d/%d frames, %d objects", n, len(idxs), len(bank))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "object_bank_val.pkl", "wb") as f:
        pickle.dump(bank, f)

    summary = {"source_frames": len(idxs), "source_segments": len(src_segs),
               "excluded_segments": len(excluded), "frame_stride": a.frame_stride,
               "max_range": a.max_range, "min_points": a.min_points, "classes": {}}
    for c in a.classes:
        objs = [o for o in bank if o["class"] == c]
        npts = np.array([o["num_points"] for o in objs]) if objs else np.zeros(0)
        rr = np.array([o["range"] for o in objs]) if objs else np.zeros(0)
        band = (rr >= 13) & (rr <= 22)   # where ATLAS placements land (15-20 m +- tol)
        summary["classes"][c] = {
            "instances": len(objs),
            "distinct_objects": len({(o["segment"], o["obj_id"]) for o in objs}),
            "instances_13_22m": int(band.sum()),
            "distinct_objects_13_22m": len({(o["segment"], o["obj_id"]) for o, b in zip(objs, band) if b}),
            "points_p10_p50_p90_13_22m": (np.percentile(npts[band], [10, 50, 90]).tolist()
                                          if band.any() else None),
        }

    rng = np.random.default_rng(a.seed)
    for spec in a.trace_band:
        parts = spec.split(":")
        cls, name, lo, hi = parts[0], parts[1], int(parts[2]), int(parts[3])
        count = int(parts[4]) if len(parts) > 4 else 300
        cand = [o for o in bank if o["class"] == cls and lo <= o["num_points"] <= hi]
        # one instance per distinct object first (diversity), then fill
        by_obj = {}
        for o in cand:
            by_obj.setdefault((o["segment"], o["obj_id"]), []).append(o)
        keys = list(by_obj)
        rng.shuffle(keys)
        chosen = [by_obj[k][int(rng.integers(len(by_obj[k])))] for k in keys[:count]]
        rest = [o for o in cand if not any(o is c for c in chosen)]
        if len(chosen) < count and rest:
            extra = rng.choice(len(rest), size=min(count - len(chosen), len(rest)), replace=False)
            chosen += [rest[int(e)] for e in extra]
        if chosen and len(chosen) < count:
            # Exactly COUNT traces: the generator's per-run placement draw uses
            # rng.integers(0, len(traces)), so every bank must have the same size
            # (300, like traces_*.pkl) for all variants to share placements.
            fill = rng.choice(len(chosen), size=count - len(chosen), replace=True)
            chosen += [chosen[int(e)] for e in fill]
        traces = [{"frame": o["frame"], "box": o["box"], "points": o["points"],
                   "num_points": o["num_points"], "segment": o["segment"],
                   "obj_id": o["obj_id"]} for o in chosen]
        out = a.out_dir / f"traces_val_{cls.lower()}_{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(traces, f)
        summary.setdefault("trace_banks", {})[out.name] = {
            "class": cls, "points_band": [lo, hi], "count": len(traces),
            "distinct_objects": len({(t["segment"], t["obj_id"]) for t in traces}),
            "candidates_in_band": len(cand)}
        if summary["trace_banks"][out.name]["distinct_objects"] < count:
            logger.warning("%s: %d distinct objects for %d traces (repeats used)", out.name,
                           summary["trace_banks"][out.name]["distinct_objects"], count)

    (a.out_dir / "bank_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
