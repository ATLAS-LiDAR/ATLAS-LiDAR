#!/usr/bin/env python
"""Scene-level perturbation datasets (spray, crosstalk, fog, temporal faults, pose noise) for ATLAS CVPR'27.

Writes the standard ATLAS dataset layout so the existing eval scripts run unchanged:
  <out>/dataset/<segment>.pkl   list of frame dicts (same keys as the ATLAS removal / baseline datasets)
  <out>/annos/<segment>_a.pkl   copied from the paired ATLAS removal reference (same targets)
  <out>/metrics/frames/<segment>_frames.csv, metrics/run_manifest.json (+ pose_noise.pkl for pose noise)

Clean points come from the OpenPCDet loaders exactly as in generate_data_adverse_baselines.py (single-frame
points + a zero time column; frame dict from the 4-frame loader). Perturbations are applied on the attack
blocks of the shared schedule (local frames 32-63, 96-127, ...), scene-wide, not firing-gated. Each frame's
spoof_gt / removal_meta (the tracked target vehicle) is copied from the paired ATLAS removal dataset
(--reference, default atlas_removal_az20_seed42), so removal-style metrics (conditional HR, collateral
damage) are computed on the same frames and targets as ATLAS removal and the point-loss baselines.
Resumable per segment.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_perturbations as SP  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--cp_mf_root", required=True)
    p.add_argument("--cp_mf_cfg", required=True)
    p.add_argument("--cp_sf_cfg", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path, help="paired ATLAS removal dataset dir")
    p.add_argument("--perturbation", required=True,
                   choices=sorted(SP.PER_FRAME) + sorted(SP.SEQUENCE) + ["pose_noise"])
    p.add_argument("--params", default="{}", help='JSON kwargs for the operator, e.g. {"alpha": 0.06}')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_sequences", type=int, default=50)
    p.add_argument("--sequence_selection", default="random", choices=("random", "first"))
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_segments", type=int, default=0, help="smoke test: stop after this many segments")
    return p.parse_args()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    a = parse()
    params = json.loads(a.params)
    np.random.seed(a.seed)   # pcdet shuffles points at test time too (shuffle_points test: True): fix the order
    for d in ("dataset", "annos", "metrics/frames"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.cp_mf_root)
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    log = common_utils.create_logger()

    def load(cfg_file):
        cfg.clear(); cfg_from_yaml_file(cfg_file, cfg); cfg.DATA_CONFIG.DATA_PATH = a.data_path
        ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
                                    dist=False, workers=a.workers, logger=log, training=False)
        return ds
    ds_mf, ds_sf = load(a.cp_mf_cfg), load(a.cp_sf_cfg)
    mf_seg = lambda i: i["point_cloud"]["lidar_sequence"]
    sf_seg = lambda i: i["frame_id"].rsplit("_", 1)[0]
    common = sorted({mf_seg(i) for i in ds_mf.infos} & {sf_seg(i) for i in ds_sf.infos})
    if a.sequence_selection == "first":
        segs = common[:a.num_sequences]
    else:   # identical to every ATLAS generator: same seed -> same 50 segments
        segs = sorted(np.random.default_rng(a.seed).choice(common, size=a.num_sequences, replace=False).tolist())
    mf_idx, sf_idx = defaultdict(list), defaultdict(list)
    for k, i in enumerate(ds_mf.infos):
        mf_idx[mf_seg(i)].append(k)
    for k, i in enumerate(ds_sf.infos):
        sf_idx[sf_seg(i)].append(k)

    man = {"generator": "gen_scene_perturb.py", "generator_sha256": sha(__file__),
           "operators_sha256": sha(SP.__file__), "perturbation": a.perturbation, "params": params,
           "seed": a.seed, "segments": segs, "reference": str(a.reference), "block": a.block,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(man, open(a.out / "metrics/run_manifest.json", "w"), indent=1)
    pose_side = {}
    for si, seg in enumerate(segs):
        if a.max_segments and si >= a.max_segments:
            break
        dst, dsa = a.out / "dataset" / f"{seg}.pkl", a.out / "annos" / f"{seg}_a.pkl"
        if dst.is_file() and dsa.is_file():
            log.info("skip %s (done)", seg); continue
        ref = pickle.load(open(a.reference / "dataset" / f"{seg}.pkl", "rb"))
        n = len(mf_idx[seg])
        if len(ref) != n or len(sf_idx[seg]) != n:
            raise RuntimeError(f"{seg}: frame count mismatch mf={n} sf={len(sf_idx[seg])} ref={len(ref)}")
        frames, pts = [], []
        for li, (gi, gs) in enumerate(zip(mf_idx[seg], sf_idx[seg])):
            fr = ds_mf[gi].copy()
            for k in ("voxels", "voxel_coords", "voxel_num_points"):
                fr.pop(k, None)
            raw = ds_sf[gs]["points"]
            p = np.hstack([raw, np.zeros((len(raw), 1), raw.dtype)]).astype(np.float32)
            fr["pose"] = fr["poses"][:4]
            if str(fr["frame_id"]) != str(ref[li]["frame_id"]):
                raise RuntimeError(f"{seg}[{li}] frame_id mismatch with reference")
            fr["spoof_gt"] = ref[li].get("spoof_gt")
            fr["removal_meta"] = ref[li].get("removal_meta")
            frames.append(fr); pts.append(p)
        attack = [li for li in range(n) if (li // a.block) % 2 == 1]
        blocks = [(b * a.block, min((b + 1) * a.block, n)) for b in range(1, (n + a.block - 1) // a.block, 2)]
        before = [len(p) for p in pts]
        metas = {}
        if a.perturbation in SP.SEQUENCE:
            pts, metas = SP.SEQUENCE[a.perturbation](pts, blocks, SP.stream(a.seed, a.perturbation, seg), **params)
        elif a.perturbation in SP.PER_FRAME:
            for li in attack:
                pts[li], metas[li] = SP.PER_FRAME[a.perturbation](
                    pts[li], frames[li], SP.stream(a.seed, a.perturbation, seg, li), **params)
        else:   # pose_noise: points unchanged; noisy poses for history alignment (sidecar + frame field)
            for li in attack:
                noisy = SP.pose_noise(frames[li]["pose"], SP.stream(a.seed, "pose_noise", seg, li), **params)
                frames[li]["pose_true"] = np.asarray(frames[li]["pose"]).copy()
                frames[li]["pose"] = noisy.astype(np.float64)
                pose_side[(seg, li)] = noisy
                metas[li] = {"fault": "pose_noise"}
        rows = []
        for li, fr in enumerate(frames):
            fr["points"] = pts[li]
            m = metas.get(li)
            fr["perturb_meta"] = None if m is None else {"perturbation": a.perturbation, **{
                k: v for k, v in m.items() if k != "clusters"}, "clusters": m.get("clusters")}
            if fr.get("removal_meta") is not None:
                fr["removal_meta"] = dict(fr["removal_meta"], n_points_before=before[li], n_points_after=len(pts[li]))
            rows.append({"frame_id": fr["frame_id"], "local_idx": li, "attack_block": li in attack,
                         "perturbed": m is not None, "n_points_before": before[li], "n_points_after": len(pts[li]),
                         "target_active": fr.get("spoof_gt") is not None,
                         **{k: v for k, v in (m or {}).items() if np.isscalar(v)}})
        tmp = dst.with_suffix(".tmp")
        pickle.dump(frames, open(tmp, "wb")); os.replace(tmp, dst)
        shutil.copy2(a.reference / "annos" / f"{seg}_a.pkl", dsa)
        keys = sorted({k for r in rows for k in r})
        with open(a.out / "metrics/frames" / f"{seg}_frames.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        log.info("[%d/%d] %s: %d frames, %d perturbed", si + 1, len(segs), seg, n, len(metas))
    if a.perturbation == "pose_noise":
        old = a.out / "metrics/pose_noise.pkl"
        if old.is_file():
            pose_side = {**pickle.load(open(old, "rb")), **pose_side}
        pickle.dump(pose_side, open(old, "wb"))
    man["status"] = "smoke" if a.max_segments else "completed"; man["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(man, open(a.out / "metrics/run_manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
