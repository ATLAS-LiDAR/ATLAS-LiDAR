#!/usr/bin/env python
"""Evaluate every KITTI model on the clean val split (Car AP@R40) and on all 9
ATLAS adversarial variants (ASR), aggregating into one CSV.

Clean metric: KITTI Car 3D AP@R40 at easy/moderate/hard (moderate is the
headline number the OpenPCDet zoo reports). Stored in the CSV's mAP column as
the moderate AP; NDS column reused for easy/hard packed as "easy|hard".

ASR:
  inject  : phantom car detected overlapping spoof_gt (IoU>=thresh)
  removal : the erased real car NOT detected (IoU<thresh)

Variants: inject_{easy,medium,hard} + removal_az{10..60} (9 total). KITTI is
single-frame so there is no global/relative injection split.

CSV columns: model, config, split, metric_type, AP_mod, AP_easy_hard, ASR,
             num_targets, num_success, notes

Run from tools/.
"""
import argparse
import csv
import os
import sys
import glob
import pickle as pkl
import traceback

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils

INJECT = ["inject_easy", "inject_medium", "inject_hard"]
REMOVAL = [f"removal_az{a}" for a in (10, 20, 30, 40, 50, 60)]

# model -> (config, checkpoint, needs_camera)
MODELS = [
    ("pointpillar",     "cfgs/kitti_models/pointpillar.yaml",     "pointpillar.pth",     False),
    ("second",          "cfgs/kitti_models/second.yaml",          "second.pth",          False),
    ("second_iou",      "cfgs/kitti_models/second_iou.yaml",      "second_iou.pth",      False),
    ("pointrcnn",       "cfgs/kitti_models/pointrcnn.yaml",        "pointrcnn.pth",       False),
    ("pointrcnn_iou",   "cfgs/kitti_models/pointrcnn_iou.yaml",    "pointrcnn_iou.pth",   False),
    ("PartA2",          "cfgs/kitti_models/PartA2.yaml",           "PartA2.pth",          False),
    ("PartA2_free",     "cfgs/kitti_models/PartA2_free.yaml",      "PartA2_free.pth",     False),
    ("pv_rcnn",         "cfgs/kitti_models/pv_rcnn.yaml",          "pv_rcnn.pth",         False),
    ("voxel_rcnn_car",  "cfgs/kitti_models/voxel_rcnn_car.yaml",   "voxel_rcnn_car.pth",  False),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--only_model", default=None)
    p.add_argument("--skip_clean", action="store_true")
    return p.parse_args()


def build(cfg_file, ckpt, logger):
    cfg.clear()
    cfg_from_yaml_file(cfg_file, cfg)
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=4, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()
    return dataset, loader, model


def eval_clean(dataset, loader, model, logger):
    det = []
    for batch in loader:
        load_data_to_gpu(batch)
        with torch.no_grad():
            pd, _ = model(batch)
        det += dataset.generate_prediction_dicts(batch, pd, cfg.CLASS_NAMES)
    ap_str, ap = dataset.evaluation(det, cfg.CLASS_NAMES)
    logger.info("\n%s", ap_str)
    # Car 3D AP@R40
    mod = ap.get("Car_3d/moderate_R40", float("nan"))
    easy = ap.get("Car_3d/easy_R40", float("nan"))
    hard = ap.get("Car_3d/hard_R40", float("nan"))
    return float(mod), float(easy), float(hard)


def infer(dataset, model, points, frame_id):
    calib = dataset.get_calib(frame_id)
    img_shape = np.array([375, 1242], dtype=np.int32)
    # use the frame's real image shape if available
    for info in dataset.kitti_infos:
        if info["point_cloud"]["lidar_idx"] == frame_id:
            img_shape = info["image"]["image_shape"]
            break
    dd = dataset.prepare_data({"points": points, "frame_id": frame_id,
                               "calib": calib, "image_shape": img_shape})
    batch = dataset.collate_batch([dd])
    load_data_to_gpu(batch)
    with torch.no_grad():
        pd, _ = model(batch)
    return dataset.generate_prediction_dicts(batch, pd, cfg.CLASS_NAMES)[0]


def eval_asr(dataset, model, split_dir, attack, iou_thresh):
    files = sorted(glob.glob(os.path.join(split_dir, "*.pkl")))
    nt = ns = 0
    for f in files:
        fr = pkl.load(open(f, "rb"))
        spoof_gt = fr.get("spoof_gt")
        if spoof_gt is None:
            continue
        nt += 1
        try:
            pred = infer(dataset, model, fr["points"][:, :4], fr["frame_id"])
        except Exception:
            # some point-based models (PointRCNN) sample a fixed #points with
            # replace=False and fail when heavy removal thins a frame below that.
            # Treat as "no detection": a removal success, an injection miss.
            if attack == "removal":
                ns += 1
            continue
        pb = pred["boxes_lidar"]
        if pb.shape[0] == 0:
            if attack == "removal":
                ns += 1
            continue
        gt = torch.tensor(np.asarray(spoof_gt[:7], dtype=np.float32)).unsqueeze(0).cuda().float()
        iou = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(pb[:, :7]).cuda().float(), gt)
        hit = iou.max().item() >= iou_thresh
        if (attack == "inject" and hit) or (attack == "removal" and not hit):
            ns += 1
    return (ns / nt if nt else float("nan")), nt, ns


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    logger = common_utils.create_logger(os.path.join(args.work_dir, "eval_kitti.log"))

    write_header = not os.path.exists(args.out_csv)
    f = open(args.out_csv, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["model", "config", "split", "metric_type", "AP_mod",
                    "AP_easy_hard", "ASR", "num_targets", "num_success", "notes"])
        f.flush()

    for name, cfg_file, ckpt_name, needs_cam in MODELS:
        if args.only_model and name != args.only_model:
            continue
        ckpt = os.path.join(args.ckpt_dir, ckpt_name)
        if needs_cam:
            w.writerow([name, cfg_file, "clean", "skipped", "", "", "", "", "", "needs camera"]); f.flush(); continue
        if not os.path.exists(ckpt):
            w.writerow([name, cfg_file, "clean", "error", "", "", "", "", "", f"missing {ckpt_name}"]); f.flush(); continue
        try:
            logger.info("===== MODEL %s =====", name)
            dataset, loader, model = build(cfg_file, ckpt, logger)
            if not args.skip_clean:
                mod, easy, hard = eval_clean(dataset, loader, model, logger)
                w.writerow([name, cfg_file, "clean", "detection", f"{mod:.4f}",
                            f"{easy:.4f}|{hard:.4f}", "", "", "", "Car_3d_R40"]); f.flush()
                logger.info("%s clean Car AP@R40 mod=%.2f", name, mod)
            for split in INJECT + REMOVAL:
                atk = "inject" if split.startswith("inject") else "removal"
                sd = os.path.join(args.atlas_dir, split)
                if not os.path.isdir(sd):
                    w.writerow([name, cfg_file, split, "asr", "", "", "", "", "", "missing dir"]); f.flush(); continue
                asr, nt, ns = eval_asr(dataset, model, sd, atk, args.iou_thresh)
                w.writerow([name, cfg_file, split, "asr", "", "", f"{asr:.4f}", nt, ns, atk]); f.flush()
                logger.info("%s %s: ASR=%.4f (%d/%d)", name, split, asr, ns, nt)
            del model, dataset, loader
            torch.cuda.empty_cache()
        except Exception:
            w.writerow([name, cfg_file, "", "error", "", "", "", "", "", traceback.format_exc().splitlines()[-1]]); f.flush()
            logger.error("MODEL %s FAILED:\n%s", name, traceback.format_exc())
    f.close()
    logger.info("wrote CSV -> %s", args.out_csv)


if __name__ == "__main__":
    main()
