#!/usr/bin/env python
"""Intensity ablation: does making ATLAS phantoms as bright as a REAL spoofer
change injection ASR?

Motivation. The OSF k6nf2 physical capture shows real injected returns saturate
the reflectivity byte (median 255/255, fraction==255 = 1.000), whereas ATLAS
samples phantom intensity from the range-conditioned DIFFUSE statistics
(median ~10-19 on nuScenes' 0-255 scale). If detectors use intensity at all,
ATLAS injection ASR is measured under an unrealistically dim phantom.

This re-runs injection ASR on the generated splits with the phantom's intensity
column overwritten, sweeping the value, while leaving geometry untouched. The
phantom points are the LAST n_spoof_k rows of `points` (spoof_frame appends
[scene_kept, spoof_kept]).

nuScenes normalises intensity into points[:,3]; we set it to the level a real
spoofer would produce. --intensity is given on the raw 0-255 byte scale and
converted using the same convention the loader uses (raw/255 if the split's
values look normalised, else raw).

CSV columns: model, split, intensity_mode, intensity_value, ASR, num_targets,
             num_success, delta_vs_baseline
"""
import argparse
import csv
import os
import pickle as pkl
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas_generation"))
import nuscenes_atlas_common as C  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.ops.iou3d_nms import iou3d_nms_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

MODELS = {
    "pp_multihead":          ("cfgs/nuscenes_models/cbgs_pp_multihead.yaml",        "pp_multihead_nds5823_updated.pth"),
    "second_multihead":      ("cfgs/nuscenes_models/cbgs_second_multihead.yaml",    "cbgs_second_multihead.pth"),
    "dyn_pp_centerpoint":    ("cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml",  "cbgs_dyn_pp_centerpoint.pth"),
    "voxel0075_voxelnext":   ("cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml", "cbgs_voxel0075_voxelnext.pth"),
    "transfusion_lidar":     ("cfgs/nuscenes_models/transfusion_lidar.yaml",        "transfusion_lidar.pth"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--models", nargs="+", default=["pp_multihead"])
    p.add_argument("--splits", nargs="+",
                   default=["inject_relative_easy", "inject_relative_medium",
                            "inject_relative_hard"])
    p.add_argument("--intensities", nargs="+", type=float,
                   default=[-1, 255],
                   help="raw 0-255 intensity to force on phantom points; "
                        "-1 = leave untouched (baseline)")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--data_path", default="../data/nuscenes")
    p.add_argument("--iou_thresh", type=float, default=0.3)
    return p.parse_args()


def build(cfg_file, ckpt, version, data_path, logger):
    cfg.clear()
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.DATA_CONFIG.VERSION = version
    cfg.DATA_CONFIG.DATA_PATH = data_path
    cfg.DATA_CONFIG.INFO_PATH = {
        "train": ["nuscenes_infos_10sweeps_train.pkl"],
        "test": ["nuscenes_infos_10sweeps_val.pkl"]}
    cfg.DATA_CONFIG.DATA_SPLIT = {"train": "train", "test": "val"}
    if "BALANCED_RESAMPLING" in cfg.DATA_CONFIG:
        cfg.DATA_CONFIG.BALANCED_RESAMPLING = False
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=4, logger=logger, training=False)
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
        pred, _ = model(batch)
    return dataset.generate_prediction_dicts(batch, pred, cfg.CLASS_NAMES)[0]


def run_split(dataset, model, split_dir, scene_to_idx, iou_thresh,
              force_intensity, logger):
    nt = ns = 0
    scale_note = None
    for scene, indices in scene_to_idx.items():
        seg_file = os.path.join(split_dir, f"{scene}_d.pkl")
        if not os.path.exists(seg_file):
            continue
        with open(seg_file, "rb") as f:
            seg = pkl.load(f)
        for local_idx, gidx in enumerate(indices):
            if local_idx >= len(seg):
                break
            frame = seg[local_idx]
            spoof_gt = frame.get("spoof_gt")
            if spoof_gt is None:
                continue
            nt += 1
            pts = np.array(frame["points"], copy=True)
            nk = int(frame.get("n_spoof_k", 0) or 0)
            if force_intensity >= 0 and nk >= 1:
                # match the scale the split actually stores intensity on
                col = pts[:, 3]
                if scale_note is None:
                    scale_note = "norm" if np.nanmax(col) <= 1.5 else "raw"
                    logger.info("intensity scale detected: %s (max %.3f)",
                                scale_note, float(np.nanmax(col)))
                val = (force_intensity / 255.0 if scale_note == "norm"
                       else force_intensity)
                pts[-nk:, 3] = val
            pred = infer(dataset, model, pts[:, :-1],
                         frame["frame_id"], frame["metadata"]["token"])
            pb = pred["boxes_lidar"]
            if pb.shape[0] == 0:
                continue
            gt = torch.tensor(np.asarray(spoof_gt[:7], dtype=np.float32)
                              ).unsqueeze(0).cuda().float()
            iou = iou3d_nms_utils.boxes_iou3d_gpu(
                torch.tensor(pb[:, :7]).cuda().float(), gt)
            if iou.max().item() >= iou_thresh:
                ns += 1
    return (ns / nt if nt else float("nan")), nt, ns


def main():
    a = parse_args()
    os.makedirs(a.work_dir, exist_ok=True)
    logger = common_utils.create_logger(os.path.join(a.work_dir, "ablate_intensity.log"))

    new = not os.path.exists(a.out_csv)
    f = open(a.out_csv, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "split", "intensity_mode", "intensity_value",
                    "ASR", "num_targets", "num_success", "delta_vs_baseline"])
        f.flush()

    from nuscenes.nuscenes import NuScenes
    for mname in a.models:
        if mname not in MODELS:
            logger.error("unknown model %s", mname)
            continue
        cfg_file, ckpt_name = MODELS[mname]
        ckpt = os.path.join(a.ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt):
            logger.error("missing ckpt %s", ckpt)
            continue
        try:
            dataset, model = build(cfg_file, ckpt, a.version, a.data_path, logger)
            nusc = NuScenes(version=a.version, dataroot=str(dataset.root_path),
                            verbose=False)
            scene_to_idx, _, _ = C.build_scene_index(dataset, nusc)
            for split in a.splits:
                sd = os.path.join(a.atlas_dir, split)
                if not os.path.isdir(sd):
                    logger.error("missing split %s", sd)
                    continue
                base = None
                for inten in a.intensities:
                    mode = "baseline" if inten < 0 else "forced"
                    asr, nt, ns = run_split(dataset, model, sd, scene_to_idx,
                                            a.iou_thresh, inten, logger)
                    if mode == "baseline":
                        base = asr
                    delta = "" if base is None else f"{asr - base:+.4f}"
                    w.writerow([mname, split, mode,
                                "" if inten < 0 else f"{inten:.0f}",
                                f"{asr:.4f}", nt, ns, delta])
                    f.flush()
                    logger.info("%s %s %s I=%s ASR=%.4f (%d/%d) delta=%s",
                                mname, split, mode, inten, asr, ns, nt, delta)
            del model, dataset
            torch.cuda.empty_cache()
        except Exception:
            logger.error("FAILED %s:\n%s", mname, traceback.format_exc())
    f.close()
    logger.info("wrote %s", a.out_csv)


if __name__ == "__main__":
    main()
