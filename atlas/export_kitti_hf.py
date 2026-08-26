#!/usr/bin/env python
"""Convert the ATLAS KITTI splits to sharded Parquet for a HuggingFace release.

Source layout (one pickle per frame, ~300 KB, 31489 files total):
    kitti/atlas_val/<split>/<frame_id>.pkl
      frame_id : str
      points   : (N, 4) float32   x, y, z, intensity  -- the ATTACKED cloud
      spoof_gt : (7,)  float64    phantom box (injection) or erased box (removal)
      anno     : dict             attack metadata (6 fields injection, 20 removal)

Output: data/<split>/train-XXXXX-of-YYYYY.parquet, one HF *config* per split.

Three deliberate choices:

1. **Clean KITTI ground truth is included** (`gt_boxes`, `gt_names`, plus
   difficulty / num_points_in_gt). The source pickles carry only `spoof_gt`, i.e.
   the attacked object, so without this the dataset can only support ASR and not
   standard mAP. Joining on frame_id against the user's own KITTI copy would work
   but invites silent misalignment against the wrong split file. Including it costs
   a few hundred bytes per frame.

   NOTE the alignment trap: annos['name'] includes 'DontCare' rows but
   annos['gt_boxes_lidar'] does NOT, so the two arrays have different lengths
   (e.g. 6 names vs 3 boxes). Names must be filtered to non-DontCare, in order,
   before zipping. Getting this wrong silently mislabels every box.

2. **The anno dict is flattened** into top-level columns, prefixed `atk_`. Nested
   structs are awkward in the HF viewer and cannot be filtered on. Injection and
   removal have disjoint schemas, which is fine because each split is its own
   config -- no null padding needed.

3. **Points are stored flat** (float32 list) plus `num_points`, rather than as a
   nested list-of-lists. Parquet handles a flat array far more efficiently, and
   `points.reshape(-1, 4)` restores it. The card documents this.

Frames absent from a split were never attacked and so were never written:
  inject_easy   3769 (full val)   inject_medium 3767   inject_hard 3649
  removal_az*   3384 (385 frames have no suitable car to erase)
ASR denominators must therefore be the split's own frame count, NOT 3769.
"""
import argparse
import glob
import os
import pickle as pkl

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ([f"inject_{d}" for d in ("easy", "medium", "hard")]
          + [f"removal_az{a}" for a in (10, 20, 30, 40, 50, 60)])

# anno fields we promote to columns, per attack family. Anything not listed is
# dropped (and reported), so the schema is explicit rather than inferred.
INJECT_ANNO = ["perturbation", "seed", "offset", "yaw", "n_spoof_r", "n_spoof_k"]
REMOVAL_ANNO = [
    "active_this_frame", "az_center_rad", "el_center_rad", "az_width_deg",
    "el_width_deg", "p_remove", "n_points_before", "n_points_in_sector",
    "n_points_removed", "n_points_after", "range_gate_passed",
    "extrapolated_close_range", "inactive_reason", "mode",
    "raw_target_az_rad", "raw_target_el_rad", "max_az_step_deg",
    "fix_elevation", "target_range_m",
]
# `target_box` is handled separately (array, not scalar)


def load_clean_gt(kitti_root):
    """{frame_id: dict} of clean KITTI GT + calibration, keyed by lidar_idx.

    Filters DontCare from `name` so it aligns with `gt_boxes_lidar`.

    Calibration is included because the official KITTI 3D AP is computed in CAMERA
    coordinates: predictions are projected with P2/R0_rect/Tr_velo_to_cam and
    clipped to the image plane. ASR only needs `spoof_gt` in the LiDAR frame, so
    the attack metrics work without calib -- but standard KITTI evaluation does
    not, and the source pickles never carried it. Cost is ~150 bytes/frame.

    All three matrices are 4x4 (OpenPCDet stores them homogeneous-padded).
    `image_shape` genuinely varies across frames (4 distinct values in val), so it
    must be stored per frame rather than assumed constant.
    """
    infos = pkl.load(open(os.path.join(kitti_root, "kitti_infos_val.pkl"), "rb"))
    out = {}
    for info in infos:
        fid = info["point_cloud"]["lidar_idx"]
        a = info.get("annos") or {}
        names = np.asarray(a.get("name", []))
        boxes = np.asarray(a.get("gt_boxes_lidar", np.zeros((0, 7))))
        keep = names != "DontCare"
        kept_names = names[keep]
        if len(kept_names) != len(boxes):
            # Defensive: if the invariant ever breaks, truncate to the shorter
            # rather than silently mispairing names with boxes.
            n = min(len(kept_names), len(boxes))
            kept_names, boxes = kept_names[:n], boxes[:n]
        diff = np.asarray(a.get("difficulty", []))[keep] if len(names) else np.array([])
        npig = np.asarray(a.get("num_points_in_gt", []))[keep] if len(names) else np.array([])
        n = len(boxes)
        cal = info.get("calib") or {}
        img = info.get("image") or {}
        out[fid] = {
            "gt_names": [str(x) for x in kept_names],
            "gt_boxes": np.asarray(boxes, dtype=np.float32).reshape(-1).tolist(),
            "num_gt": n,
            "gt_difficulty": [int(x) for x in diff[:n]] if len(diff) else [],
            "gt_num_points": [int(x) for x in npig[:n]] if len(npig) else [],
            # calibration, flattened 4x4 row-major
            "calib_P2": np.asarray(cal.get("P2"), dtype=np.float32).reshape(-1).tolist(),
            "calib_R0_rect": np.asarray(cal.get("R0_rect"), dtype=np.float32).reshape(-1).tolist(),
            "calib_Tr_velo_to_cam": np.asarray(cal.get("Tr_velo_to_cam"), dtype=np.float32).reshape(-1).tolist(),
            "image_shape": [int(x) for x in np.asarray(
                img.get("image_shape", [375, 1242])).reshape(-1)],
        }
    return out


def rows_for_split(split_dir, clean_gt, family, limit=None):
    files = sorted(glob.glob(os.path.join(split_dir, "*.pkl")))
    if limit:
        files = files[:limit]
    unknown = set()
    for f in files:
        try:
            d = pkl.load(open(f, "rb"))
        except Exception:
            continue
        fid = str(d.get("frame_id") or os.path.basename(f)[:-4])
        P = np.asarray(d["points"], dtype=np.float32)
        sg = np.asarray(d.get("spoof_gt"), dtype=np.float32).reshape(-1)
        anno = d.get("anno") or {}

        row = {
            "frame_id": fid,
            "points": P.reshape(-1).tolist(),
            "num_points": int(P.shape[0]),
            "point_dim": int(P.shape[1]),
            "spoof_gt": sg.tolist(),
        }
        # clean KITTI GT (self-containment)
        g = clean_gt.get(fid)
        if g is None:
            # should not happen (every split frame comes from val), but keep the
            # schema total rather than emitting a row with missing columns
            row.update(gt_names=[], gt_boxes=[], num_gt=0,
                       gt_difficulty=[], gt_num_points=[],
                       calib_P2=[], calib_R0_rect=[], calib_Tr_velo_to_cam=[],
                       image_shape=[])
            print(f"    [warn] no KITTI info for frame {fid}")
        else:
            row.update(g)

        keys = INJECT_ANNO if family == "inject" else REMOVAL_ANNO
        for k in keys:
            v = anno.get(k)
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v, dtype=np.float64).reshape(-1).tolist()
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            row[f"atk_{k}"] = v
        if family == "removal":
            tb = anno.get("target_box")
            row["atk_target_box"] = (np.asarray(tb, dtype=np.float32).reshape(-1).tolist()
                                     if tb is not None else [])
        unknown |= (set(anno) - set(keys) - {"target_box"})
        yield row
    if unknown:
        print(f"    [note] anno fields NOT exported: {sorted(unknown)}")


def build_schema(family):
    """Explicit schema. Without this, pyarrow infers float64 from Python lists and
    the point arrays double in size for no benefit -- the source is float32."""
    f32, f64, i64 = pa.float32(), pa.float64(), pa.int64()
    fields = [
        ("frame_id", pa.string()),
        ("points", pa.list_(f32)),
        ("num_points", i64),
        ("point_dim", i64),
        ("spoof_gt", pa.list_(f32)),
        ("gt_names", pa.list_(pa.string())),
        ("gt_boxes", pa.list_(f32)),
        ("num_gt", i64),
        ("gt_difficulty", pa.list_(i64)),
        ("gt_num_points", pa.list_(i64)),
        ("calib_P2", pa.list_(f32)),
        ("calib_R0_rect", pa.list_(f32)),
        ("calib_Tr_velo_to_cam", pa.list_(f32)),
        ("image_shape", pa.list_(i64)),
    ]
    if family == "inject":
        fields += [
            ("atk_perturbation", pa.string()), ("atk_seed", i64),
            ("atk_offset", pa.list_(f64)), ("atk_yaw", f64),
            ("atk_n_spoof_r", i64), ("atk_n_spoof_k", i64),
        ]
    else:
        fields += [
            ("atk_active_this_frame", pa.bool_()), ("atk_az_center_rad", f64),
            ("atk_el_center_rad", f64), ("atk_az_width_deg", i64),
            ("atk_el_width_deg", f64), ("atk_p_remove", f64),
            ("atk_n_points_before", i64), ("atk_n_points_in_sector", i64),
            ("atk_n_points_removed", i64), ("atk_n_points_after", i64),
            ("atk_range_gate_passed", pa.bool_()),
            ("atk_extrapolated_close_range", pa.bool_()),
            ("atk_inactive_reason", pa.string()), ("atk_mode", pa.string()),
            ("atk_raw_target_az_rad", f64), ("atk_raw_target_el_rad", f64),
            ("atk_max_az_step_deg", f64), ("atk_fix_elevation", pa.bool_()),
            ("atk_target_range_m", f64), ("atk_target_box", pa.list_(f32)),
        ]
    return pa.schema(fields)


def write_split(split, src_root, out_root, clean_gt, rows_per_shard, limit=None):
    family = "inject" if split.startswith("inject") else "removal"
    split_dir = os.path.join(src_root, split)
    if not os.path.isdir(split_dir):
        print(f"  [skip] {split}: not found")
        return 0
    out_dir = os.path.join(out_root, "data", split)
    os.makedirs(out_dir, exist_ok=True)
    schema = build_schema(family)

    buf, shards, total = [], [], 0
    def flush():
        nonlocal buf
        if not buf:
            return
        tbl = pa.Table.from_pylist(buf, schema=schema)
        shards.append(tbl)
        buf = []

    for row in rows_for_split(split_dir, clean_gt, family, limit):
        buf.append(row)
        total += 1
        if len(buf) >= rows_per_shard:
            flush()
    flush()

    n = len(shards)
    for i, tbl in enumerate(shards):
        path = os.path.join(out_dir, f"train-{i:05d}-of-{n:05d}.parquet")
        pq.write_table(tbl, path, compression="zstd")
    size = sum(os.path.getsize(os.path.join(out_dir, f))
               for f in os.listdir(out_dir)) / 1e9
    print(f"  {split:16s} {total:5d} frames -> {n} shard(s), {size:.2f} GB")
    return total


def verify(split, src_root, out_root, clean_gt, n_check=5):
    """Round-trip: Parquet rows must reproduce the source pickles exactly,
    and the added calib/GT must match kitti_infos_val.pkl."""
    import pyarrow.dataset as ds
    d = ds.dataset(os.path.join(out_root, "data", split), format="parquet")
    tbl = d.head(n_check)
    ok = True
    for row in tbl.to_pylist():
        fid = row["frame_id"]
        src = pkl.load(open(os.path.join(src_root, split, f"{fid}.pkl"), "rb"))
        P0 = np.asarray(src["points"], dtype=np.float32)
        P1 = np.asarray(row["points"], dtype=np.float32).reshape(
            row["num_points"], row["point_dim"])
        same_pts = P0.shape == P1.shape and np.array_equal(P0, P1)
        s0 = np.asarray(src["spoof_gt"], dtype=np.float32).reshape(-1)
        s1 = np.asarray(row["spoof_gt"], dtype=np.float32)
        same_gt = np.array_equal(s0, s1)

        g = clean_gt.get(fid, {})
        same_cal = all(
            np.array_equal(np.asarray(row[k], dtype=np.float32),
                           np.asarray(g.get(k, []), dtype=np.float32))
            for k in ("calib_P2", "calib_R0_rect", "calib_Tr_velo_to_cam"))
        same_img = list(row["image_shape"]) == list(g.get("image_shape", []))
        same_box = np.array_equal(
            np.asarray(row["gt_boxes"], dtype=np.float32),
            np.asarray(g.get("gt_boxes", []), dtype=np.float32))

        if not (same_pts and same_gt and same_cal and same_img and same_box):
            ok = False
            print(f"    MISMATCH {split}/{fid}: points={same_pts} spoof_gt={same_gt} "
                  f"calib={same_cal} image_shape={same_img} gt_boxes={same_box}")
    print(f"  {split:16s} round-trip {'OK' if ok else 'FAILED'} "
          f"({n_check} frames; points, spoof_gt, calib, image_shape, gt_boxes)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root",
                    default="/home/mellonzhang/mmt-nas-1/mellon/data/datasets/kitti")
    ap.add_argument("--out", required=True, help="output repo dir")
    ap.add_argument("--rows_per_shard", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None,
                    help="frames per split (smoke test)")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    a = ap.parse_args()

    src_root = os.path.join(a.kitti_root, "atlas_val")
    os.makedirs(a.out, exist_ok=True)
    print("loading clean KITTI GT from kitti_infos_val.pkl ...")
    clean_gt = load_clean_gt(a.kitti_root)
    print(f"  {len(clean_gt)} val frames, "
          f"median {np.median([v['num_gt'] for v in clean_gt.values()]):.0f} objects/frame")

    print("\nwriting Parquet:")
    for s in a.splits:
        write_split(s, src_root, a.out, clean_gt, a.rows_per_shard, a.limit)

    print("\nverifying round-trip:")
    allok = all(verify(s, src_root, a.out, clean_gt) for s in a.splits
                if os.path.isdir(os.path.join(a.out, "data", s)))
    print(f"\n{'ALL ROUND-TRIPS OK' if allok else 'SOME ROUND-TRIPS FAILED'}")


if __name__ == "__main__":
    main()
