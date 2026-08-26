#!/usr/bin/env python
"""Cache per-frame detector predictions for the nuScenes injection splits.

The ATLAS eval matrix computes IoU against `spoof_gt` inline and throws every
prediction away, so the diagnostics in atlas_eval_metrics_summary.pdf -- box
displacement and spoof-region false positives in particular -- have nothing to run
over. Those two need the full predicted box set on both the ATTACKED and the CLEAN
version of the same frame, which no existing artifact contains.

This script runs inference once per (model, config) and caches predictions, so the
metrics can be recomputed later without paying for inference again.

Two things worth noting about what gets cached:

  * Predictions are kept in the LIDAR frame (`boxes_lidar`), matching `spoof_gt`.
    The one clean artifact that already exists (eval_work/results_nusc.json) is in
    nuScenes GLOBAL coordinates and covers a single model, so reusing it would mean
    a per-frame pose transform and a model mismatch. Re-running clean inference is
    ~20 min total and avoids both.
  * Only ATTACKED frames are cached (those with a non-null `spoof_gt`), plus the
    same frames from the clean pass. Clean frames elsewhere in a scene are not
    used by any of the five metrics.

Output layout:
    <out>/<model>/<config>/<scene>_p.pkl      attacked predictions
    <out>/<model>/clean/<scene>_p.pkl         clean predictions, same frames
    <out>/<model>/<config>/_meta.json         frame count, timing, ckpt
"""
import argparse
import json
import os
import pickle as pkl
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "atlas_generation"))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
import nuscenes_atlas_common as C  # noqa: E402

INJECT_CONFIGS = [f"inject_{m}_{d}" for m in ("global", "relative")
                  for d in ("easy", "medium", "hard")]

# model -> (config relative to tools/, checkpoint filename)
MODELS = [
    ("pp_multihead", "cfgs/nuscenes_models/cbgs_pp_multihead.yaml",
     "pp_multihead_nds5823_updated.pth"),
    ("second_multihead", "cfgs/nuscenes_models/cbgs_second_multihead.yaml",
     "cbgs_second_multihead.pth"),
    ("dyn_pp_centerpoint", "cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml",
     "cbgs_dyn_pp_centerpoint.pth"),
    ("voxel0075_voxelnext", "cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml",
     "cbgs_voxel0075_voxelnext.pth"),
    ("transfusion_lidar", "cfgs/nuscenes_models/transfusion_lidar.yaml",
     "transfusion_lidar.pth"),
]


def build(cfg_file, ckpt, version, data_path, logger):
    cfg.clear()
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.DATA_CONFIG.VERSION = version
    cfg.DATA_CONFIG.DATA_PATH = data_path
    cfg.DATA_CONFIG.INFO_PATH = {
        "train": ["nuscenes_infos_10sweeps_train.pkl"],
        "test": ["nuscenes_infos_10sweeps_val.pkl"],
    }
    cfg.DATA_CONFIG.DATA_SPLIT = {"train": "train", "test": "val"}
    if "BALANCED_RESAMPLING" in cfg.DATA_CONFIG:
        cfg.DATA_CONFIG.BALANCED_RESAMPLING = False
    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=2, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                         dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()
    return dataset, model


def infer(dataset, model, points5, frame_id, token):
    dd = dataset.prepare_data(data_dict={
        "points": points5, "frame_id": frame_id, "metadata": {"token": token}})
    batch = dataset.collate_batch([dd])
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred_dicts, _ = model(batch)
    return dataset.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)[0]


def slim(anno):
    """Keep only what the metrics need; drop nothing else of substance."""
    return {
        "name": np.asarray(anno["name"]),
        "score": np.asarray(anno["score"], dtype=np.float32),
        "boxes_lidar": np.asarray(anno["boxes_lidar"], dtype=np.float32),
        "pred_labels": np.asarray(anno["pred_labels"], dtype=np.int64),
        "frame_id": str(anno["frame_id"]),
        "token": str((anno.get("metadata") or {}).get("token", "")),
    }


def run_config(dataset, model, split_dir, scene_to_idx, out_dir, logger,
               clean_out=None):
    """Cache predictions for every ATTACKED frame of one config.

    When `clean_out` is given, the SAME frames are also run on the unmodified
    cloud loaded from the dataset, giving the per-model clean baseline that the
    displacement and false-positive metrics compare against.
    """
    os.makedirs(out_dir, exist_ok=True)
    if clean_out:
        os.makedirs(clean_out, exist_ok=True)
    n_frames = n_scenes = 0
    t0 = time.time()
    for scene, indices in scene_to_idx.items():
        seg_file = os.path.join(split_dir, f"{scene}_d.pkl")
        if not os.path.exists(seg_file):
            continue
        done = os.path.exists(os.path.join(out_dir, f"{scene}_p.pkl"))
        need_clean = clean_out and not os.path.exists(
            os.path.join(clean_out, f"{scene}_p.pkl"))
        if done and not need_clean:
            continue
        with open(seg_file, "rb") as f:
            seg = pkl.load(f)

        atk_recs, clean_recs = [], []
        for local_idx, gidx in enumerate(indices):
            fr = seg[local_idx]
            if fr.get("spoof_gt") is None:
                continue
            tok = (fr.get("metadata") or {}).get("token", "")
            rec = slim(infer(dataset, model, fr["points"][:, :-1],
                             fr["frame_id"], tok))
            # carry the attack ground truth alongside, so the metrics script needs
            # only the pred cache plus the split's spoof_gt
            rec["spoof_gt"] = np.asarray(fr["spoof_gt"], dtype=np.float32)
            rec["gt_boxes"] = np.asarray(fr.get("gt_boxes"), dtype=np.float32)
            rec["frame_index"] = int(local_idx)
            atk_recs.append(rec)

            if need_clean:
                pts, _ = C.load_scene_points(dataset, gidx)
                crec = slim(infer(dataset, model, pts[:, :-1],
                                  fr["frame_id"], tok))
                crec["frame_index"] = int(local_idx)
                clean_recs.append(crec)
            n_frames += 1

        if atk_recs and not done:
            with open(os.path.join(out_dir, f"{scene}_p.pkl"), "wb") as f:
                pkl.dump(atk_recs, f)
        if clean_recs:
            with open(os.path.join(clean_out, f"{scene}_p.pkl"), "wb") as f:
                pkl.dump(clean_recs, f)
        n_scenes += 1
        if n_scenes % 25 == 0:
            logger.info("    %d scenes, %d frames (%.0fs)", n_scenes, n_frames,
                        time.time() - t0)
    return n_frames, n_scenes, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas_dir", required=True, help="atlas_val root")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--data_path", default="../data/nuscenes")
    ap.add_argument("--configs", nargs="+", default=INJECT_CONFIGS)
    ap.add_argument("--only_model", default=None)
    ap.add_argument("--skip_clean", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    logger = common_utils.create_logger(os.path.join(a.out_dir, "cache_preds.log"))

    for name, cfg_file, ckpt_name in MODELS:
        if a.only_model and name != a.only_model:
            continue
        ckpt = os.path.join(a.ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt):
            logger.info("SKIP %s: checkpoint missing (%s)", name, ckpt_name)
            continue
        logger.info("===== MODEL %s =====", name)
        try:
            dataset, model = build(cfg_file, ckpt, a.version, a.data_path, logger)
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version=a.version, dataroot=str(dataset.root_path),
                            verbose=False)
            scene_to_idx, _, _ = C.build_scene_index(dataset, nusc)

            clean_dir = None if a.skip_clean else os.path.join(a.out_dir, name, "clean")
            for i, conf in enumerate(a.configs):
                split_dir = os.path.join(a.atlas_dir, conf)
                if not os.path.isdir(split_dir):
                    logger.info("  [skip] %s: split dir missing", conf)
                    continue
                out_dir = os.path.join(a.out_dir, name, conf)
                # the clean pass only needs to happen once per model
                nf, ns, dt = run_config(
                    dataset, model, split_dir, scene_to_idx, out_dir, logger,
                    clean_out=clean_dir if i == 0 else None)
                logger.info("  %-24s %4d scenes %5d frames  %.1f min",
                            conf, ns, nf, dt / 60)
                json.dump({"model": name, "config": conf, "n_frames": nf,
                           "n_scenes": ns, "seconds": dt, "ckpt": ckpt_name},
                          open(os.path.join(out_dir, "_meta.json"), "w"), indent=1)
            del model, dataset
            torch.cuda.empty_cache()
        except Exception:
            import traceback
            logger.error("MODEL %s FAILED:\n%s", name, traceback.format_exc())


if __name__ == "__main__":
    main()
