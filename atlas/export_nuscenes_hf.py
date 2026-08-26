#!/usr/bin/env python
"""Convert the ATLAS nuScenes splits to sharded Parquet for a HuggingFace release.

Source layout: one pickle per SCENE (40 frames), plus an annos sidecar.
    atlas_val/<split>/<scene>_d.pkl        list[frame_dict]
    atlas_val/<split>_annos/<scene>_a.pkl  {segment, spoof_annos, run_annos?}

WHAT IS EXPORTED, AND WHY ONLY THE ATTACKED FRAMES
--------------------------------------------------
A nuScenes scene is ~40 frames, but the schedule attacks only the last 8, so
80-90% of every split is a bit-identical copy of source nuScenes. Shipping all
frames costs ~242 GB across 12 splits, of which ~206 GB is twelve duplicate copies
of the same clean clouds. Exporting only the attacked frames costs ~34 GB.

This loses nothing: ASR is defined only on attacked frames (both eval harnesses
skip `spoof_gt is None`), and the omitted frames are unmodified, so a user
reconstructs a full sequence by substituting our rows into their own nuScenes copy.
Every row therefore carries four independent keys back to the source:

    segment       scene name, e.g. "scene-0003"
    frame_index   position within the scene (0-based)
    sample_token  nuScenes sample token -- the canonical global key
    frame_id      source LIDAR_TOP filename

`sample_token` is authoritative: joining on it does not depend on reproducing our
scene ordering or naming. `frame_index` is for convenience and ordering.

SHARDING
--------
Rows are frames; `segment` is a grouping column. Shards are cut on SEGMENT
boundaries, so a scene is never split across files and `filter(segment == X)`
touches one shard. Two rejected alternatives:
  * one row per segment -> ~202 MB/row. Parquet reads rows atomically, so this
    breaks memory, streaming and the dataset viewer.
  * one file per segment -> attacked frames per segment range 0-8, giving 150
    mostly-tiny files per split (1800 across 12 splits).

SCHEMA NOTES
------------
* points are (N, 6) = x, y, z, intensity, time, lag, stored FLAT (float32) with
  `num_points`/`point_dim`; Parquet handles a flat array far better than nested
  lists. nuScenes clouds are 10-sweep accumulated, hence ~265k points/frame.
* `spoof_gt` is 10 columns for REMOVAL (a real GT box: 7 geometry + 2 velocity +
  class) but 7 for INJECTION (a phantom: geometry only). Stored at native length
  with `spoof_gt_dim` recording which.
* `gt_boxes` is (M, 10) with the 1-indexed class label in the last column.
"""
import argparse
import glob
import os
import pickle as pkl

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

INJECT_SPLITS = [f"inject_{m}_{d}" for m in ("global", "relative")
                 for d in ("easy", "medium", "hard")]
REMOVAL_SPLITS = [f"removal_az{a}" for a in (10, 20, 30, 40, 50, 60)]
SPLITS = INJECT_SPLITS + REMOVAL_SPLITS

INJECT_ANNO = ["perturbation", "seed", "trace_idx", "yaw"]
REMOVAL_ANNO = [
    "active_this_frame", "az_center_rad", "el_center_rad", "az_width_deg",
    "el_width_deg", "p_remove", "n_points_before", "n_points_in_sector",
    "n_points_removed", "n_points_after", "range_gate_passed",
    "extrapolated_close_range", "inactive_reason", "mode",
    "raw_target_az_rad", "raw_target_el_rad", "max_az_step_deg",
    "fix_elevation", "target_range_m",
]


# Schedule variants. The legacy schedule attacks the last 8 frames of every scene;
# these attack ONE span per scene of varying length, either contiguous (_cont) or
# flickering with interior gaps (_intermit). Config names are the legacy names plus
# a suffix, so already-published config names are never renamed.
SCHED_SUFFIXES = ["_cont", "_intermit"]
SCHED_SPLITS = [s + x for x in SCHED_SUFFIXES for s in (INJECT_SPLITS + REMOVAL_SPLITS)]

# Per-scene schedule bookkeeping, exported as sched_* columns on scheduled configs
# only. Without these you cannot stratify by attack length or separate a scheduled
# gap from a firing-gate dropout.
SCHED_META = ["mode", "attack_len_L", "span_start_local", "span_len", "n_gaps",
              "realised_duty", "degenerate", "clean_prefix", "sched_seed"]


def build_schema(family, scheduled=False):
    f32, f64, i64 = pa.float32(), pa.float64(), pa.int64()
    fields = [
        ("segment", pa.string()),
        ("frame_index", i64),
        ("num_frames_in_segment", i64),
        ("sample_token", pa.string()),
        ("frame_id", pa.string()),
        ("points", pa.list_(f32)),
        ("num_points", i64),
        ("point_dim", i64),
        ("spoof_gt", pa.list_(f32)),
        ("spoof_gt_dim", i64),
        ("gt_boxes", pa.list_(f32)),
        ("num_gt", i64),
        ("gt_box_dim", i64),
        ("pose", pa.list_(f64)),
    ]
    if family == "inject":
        fields += [
            ("atk_perturbation", pa.string()), ("atk_seed", i64),
            ("atk_trace_idx", i64), ("atk_yaw", f64),
            ("atk_offset", pa.list_(f64)),
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
            ("atk_num_tracked", i64), ("atk_total_span_frames", i64),
        ]
    if scheduled:
        fields += [
            ("sched_mode", pa.string()),
            ("sched_attack_len_L", i64),
            ("sched_span_start_local", i64),
            ("sched_span_len", i64),
            ("sched_n_gaps", i64),
            ("sched_realised_duty", f64),
            ("sched_degenerate", pa.bool_()),
            ("sched_clean_prefix", i64),
            ("sched_seed", i64),
            ("sched_attacked_frames", pa.list_(i64)),
            ("sched_gap_frames", pa.list_(i64)),
            ("scheduled_attack", pa.bool_()),
        ]
    return pa.schema(fields)


def _scalar(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def segment_rows(seg_file, annos_file, family, scheduled=False):
    """Yield one row per ATTACKED frame of this scene."""
    seg = pkl.load(open(seg_file, "rb"))
    scene = os.path.basename(seg_file)[:-len("_d.pkl")]
    annos = {}
    run_diag = {}
    sched_meta = {}
    if os.path.exists(annos_file):
        an = pkl.load(open(annos_file, "rb"))
        sa = an.get("spoof_annos") or {}
        if isinstance(sa, dict):
            annos = sa
        ra = an.get("run_annos") or []
        if ra and isinstance(ra[0], dict):
            td = ra[0].get("tracking_diagnostics") or {}
            run_diag = {"num_tracked": td.get("num_tracked"),
                        "total_span_frames": td.get("total_frames")}
        sched_meta = an.get("schedule_meta") or {}

    for idx, fr in enumerate(seg):
        if fr.get("spoof_gt") is None:
            continue
        P = np.asarray(fr["points"], dtype=np.float32)
        sg = np.asarray(fr["spoof_gt"], dtype=np.float32).reshape(-1)
        gb = np.asarray(fr.get("gt_boxes"), dtype=np.float32)
        gb = gb.reshape(-1, gb.shape[-1]) if gb.ndim == 2 else gb.reshape(0, 10)
        pose = np.asarray(fr.get("pose"), dtype=np.float64).reshape(-1)

        row = {
            "segment": scene,
            "frame_index": int(idx),
            "num_frames_in_segment": len(seg),
            "sample_token": str((fr.get("metadata") or {}).get("token", "")),
            "frame_id": str(fr.get("frame_id", "")),
            "points": P.reshape(-1).tolist(),
            "num_points": int(P.shape[0]),
            "point_dim": int(P.shape[1]),
            "spoof_gt": sg.tolist(),
            "spoof_gt_dim": int(sg.size),
            "gt_boxes": gb.reshape(-1).tolist(),
            "num_gt": int(gb.shape[0]),
            "gt_box_dim": int(gb.shape[1]) if gb.size else 10,
            "pose": pose.tolist(),
        }

        meta = annos.get(idx) if isinstance(annos.get(idx), dict) else {}
        if family == "inject":
            for k in INJECT_ANNO:
                row[f"atk_{k}"] = _scalar(meta.get(k))
            off = meta.get("relative_position_offset")
            row["atk_offset"] = ([float(x) for x in np.asarray(off).reshape(-1)]
                                 if off is not None else [])
            row["atk_n_spoof_r"] = int(fr.get("n_spoof_r") or 0)
            row["atk_n_spoof_k"] = int(fr.get("n_spoof_k") or 0)
        else:
            for k in REMOVAL_ANNO:
                row[f"atk_{k}"] = _scalar(meta.get(k))
            tb = meta.get("target_box")
            row["atk_target_box"] = (
                np.asarray(tb, dtype=np.float32).reshape(-1).tolist()
                if tb is not None else [])
            row["atk_num_tracked"] = run_diag.get("num_tracked")
            row["atk_total_span_frames"] = run_diag.get("total_span_frames")

        if scheduled:
            for k in SCHED_META:
                row[f"sched_{k}"] = _scalar(sched_meta.get(k))
            row["sched_attacked_frames"] = [int(x) for x in
                                            (sched_meta.get("attacked_local") or [])]
            row["sched_gap_frames"] = [int(x) for x in
                                       (sched_meta.get("gap_local") or [])]
            # scheduled != realised: the A-HFR firing gate and occlusion filtering
            # can drop a frame the schedule marked. Keep both so a gate dropout is
            # never mistaken for a scheduled gap.
            row["scheduled_attack"] = bool(fr.get("scheduled_attack", True))
        yield row


def write_split(split, src_root, out_root, target_shard_gb=0.45, limit=None):
    family = "inject" if split.startswith("inject") else "removal"
    scheduled = split.endswith(tuple(SCHED_SUFFIXES))
    split_dir = os.path.join(src_root, split)
    annos_dir = os.path.join(src_root, split + "_annos")
    if not os.path.isdir(split_dir):
        print(f"  [skip] {split}: not found")
        return 0, 0
    out_dir = os.path.join(out_root, "data", split)
    os.makedirs(out_dir, exist_ok=True)
    schema = build_schema(family, scheduled)

    seg_files = sorted(glob.glob(os.path.join(split_dir, "*_d.pkl")))
    if limit:
        seg_files = seg_files[:limit]

    # Group whole segments into shards so a scene never straddles a file.
    # ~6 attacked frames/scene x ~3.3 MB/frame -> ~20 MB/scene, so ~22 scenes
    # lands near a 450 MB shard.
    budget = int(max(1, target_shard_gb * 1e9 // (7 * 3.4e6)))
    shards, buf, n_rows, n_segs = [], [], 0, 0
    for sf in seg_files:
        af = os.path.join(annos_dir,
                          os.path.basename(sf).replace("_d.pkl", "_a.pkl"))
        rows = list(segment_rows(sf, af, family, scheduled))
        n_segs += 1
        if rows:
            buf.extend(rows)
            n_rows += len(rows)
        if n_segs % budget == 0 and buf:
            shards.append(buf)
            buf = []
    if buf:
        shards.append(buf)

    n = len(shards)
    total_bytes = 0
    for i, rows in enumerate(shards):
        path = os.path.join(out_dir, f"train-{i:05d}-of-{n:05d}.parquet")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path,
                       compression="zstd")
        total_bytes += os.path.getsize(path)
    print(f"  {split:26s} {n_segs:4d} segments -> {n_rows:5d} attacked frames, "
          f"{n} shard(s), {total_bytes/1e9:.2f} GB", flush=True)
    return n_rows, n_segs


def verify(split, src_root, out_root, n_check=4):
    """Round-trip a few rows against the source pickles."""
    import pyarrow.dataset as ds
    d = ds.dataset(os.path.join(out_root, "data", split), format="parquet")
    rows = d.head(n_check).to_pylist()
    ok = True
    for r in rows:
        sf = os.path.join(src_root, split, f"{r['segment']}_d.pkl")
        seg = pkl.load(open(sf, "rb"))
        fr = seg[r["frame_index"]]
        P0 = np.asarray(fr["points"], dtype=np.float32)
        P1 = np.asarray(r["points"], dtype=np.float32).reshape(
            r["num_points"], r["point_dim"])
        same_pts = P0.shape == P1.shape and np.array_equal(P0, P1)
        same_sg = np.array_equal(
            np.asarray(fr["spoof_gt"], dtype=np.float32).reshape(-1),
            np.asarray(r["spoof_gt"], dtype=np.float32))
        same_tok = (fr.get("metadata") or {}).get("token", "") == r["sample_token"]
        same_pose = np.allclose(
            np.asarray(fr["pose"], dtype=np.float64).reshape(-1),
            np.asarray(r["pose"], dtype=np.float64), rtol=0, atol=0)
        if not (same_pts and same_sg and same_tok and same_pose):
            ok = False
            print(f"    MISMATCH {split}/{r['segment']}#{r['frame_index']}: "
                  f"points={same_pts} spoof_gt={same_sg} token={same_tok} "
                  f"pose={same_pose}")
    print(f"  {split:26s} round-trip {'OK' if ok else 'FAILED'} "
          f"({len(rows)} frames; points, spoof_gt, sample_token, pose)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas_dir", default="/home/mellonzhang/mmt-nas-1/mellon/"
                    "data/datasets/nuscenes_trainval/atlas_val",
                    help="source root for the LEGACY splits")
    ap.add_argument("--sched_dir", default="/home/mellonzhang/mmt-nas-1/mellon/"
                    "data/datasets/nuscenes_trainval/atlas_val_sched",
                    help="source root for the _cont / _intermit splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=SPLITS,
                    help=f"default = the 12 legacy splits; pass 'sched' for the "
                         f"24 schedule variants, or 'all' for all 36")
    ap.add_argument("--limit", type=int, default=None,
                    help="segments per split (smoke test)")
    ap.add_argument("--shard_gb", type=float, default=0.45)
    a = ap.parse_args()

    splits = list(a.splits)
    if splits == ["sched"]:
        splits = SCHED_SPLITS
    elif splits == ["all"]:
        splits = SPLITS + SCHED_SPLITS

    def src_of(split):
        """Scheduled variants live under a different root than the legacy splits."""
        return a.sched_dir if split.endswith(tuple(SCHED_SUFFIXES)) else a.atlas_dir

    os.makedirs(a.out, exist_ok=True)
    print("writing Parquet (ATTACKED frames only):")
    tot = 0
    for s in splits:
        n, _ = write_split(s, src_of(s), a.out, a.shard_gb, a.limit)
        tot += n
    print(f"\ntotal attacked frames: {tot}")

    print("\nverifying round-trip:")
    allok = all(verify(s, src_of(s), a.out) for s in splits
                if os.path.isdir(os.path.join(a.out, "data", s)))
    print(f"\n{'ALL ROUND-TRIPS OK' if allok else 'SOME ROUND-TRIPS FAILED'}")


if __name__ == "__main__":
    main()
