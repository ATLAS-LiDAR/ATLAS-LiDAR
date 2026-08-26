#!/usr/bin/env python
"""Evaluate a detector against the nuScenes ATLAS adversarial splits (ASR).

nuScenes analogue of point_injection.py. Reads the per-scene `<scene>_d.pkl`
files produced by generate_data_{global,relative,removal}_nuscenes.py, runs the
model on the attacked point clouds, and computes an Attack Success Rate.

Two attack semantics (select with --attack_type):
  inject  : ASR = fraction of target frames where the model reports a detection
            overlapping the phantom box `spoof_gt` (a false positive was
            successfully injected).
  removal : ASR = fraction of target frames where the erased real object
            `spoof_gt` is NO LONGER detected (a true positive was removed).

Differences from the Waymo original:
  * segments are nuScenes scenes (via scene tokens), not `lidar_sequence`;
    frame<->file alignment uses the same build_scene_index the generators use.
  * inference feeds spoofed points through prepare_data WITHOUT gt_boxes (pure
    forward pass), avoiding the Waymo double-prepare gt-column mangling.
  * points are (N,6)=[x,y,z,intensity,time,lag]; [:, :-1] recovers the model's
    5-feature cloud.
  * ASR is reported over nuScenes classes; the phantom trace class is 'car'.

Example:
  python point_injection_nuscenes.py \
      --dataset  <atlas_mini>/inject_relative_easy \
      --cfg_file tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
      --ckpt     <.../pp_multihead_nds5823_updated.pth> \
      --attack_type inject \
      --pred_dir <out>/preds_relative_easy \
      --asr_path <out>/asr_relative_easy.pkl
"""
import argparse
import json
import logging
import os
import pickle as pkl
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas_generation"))
import nuscenes_atlas_common as C  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.ops.iou3d_nms import iou3d_nms_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

# nuScenes cbgs_pp_multihead class order (1-indexed labels)
VEHICLE_LABELS = {1, 2, 3, 4, 5}   # car, truck, construction_vehicle, bus, trailer
CAR_LABEL = 1


def setup_logger(log_path):
    logger = logging.getLogger("asr_eval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dir of <scene>_d.pkl attacked splits")
    p.add_argument("--cfg_file", required=True, help="model config yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attack_type", required=True, choices=["inject", "removal"])
    p.add_argument("--pred_dir", required=True, help="per-scene prediction output dir")
    p.add_argument("--asr_path", required=True, help="output ASR summary .pkl")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--data_path", default="../data/nuscenes",
                   help="DATA_PATH relative to tools/ (as OpenPCDet expects)")
    p.add_argument("--iou_thresh", type=float, default=0.3,
                   help="3D IoU to count a phantom hit / real miss")
    p.add_argument("--infer_all", action="store_true",
                   help="run inference on every frame and dump per-scene preds "
                        "(default: only target frames, which is all ASR needs)")
    p.add_argument("--log_file", default="nuscenes_asr.log")
    return p.parse_args()


def build_model_and_data(args, logger):
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    # retarget the model's data config at the mini split
    cfg.DATA_CONFIG.VERSION = args.version
    cfg.DATA_CONFIG.DATA_PATH = args.data_path
    cfg.DATA_CONFIG.INFO_PATH = {
        "train": ["nuscenes_infos_10sweeps_train.pkl"],
        "test": ["nuscenes_infos_10sweeps_val.pkl"],
    }
    # eval on the val split; no balanced resampling
    cfg.DATA_CONFIG.DATA_SPLIT = {"train": "train", "test": "val"}
    if "BALANCED_RESAMPLING" in cfg.DATA_CONFIG:
        cfg.DATA_CONFIG.BALANCED_RESAMPLING = False

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=1, logger=logger, training=False)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                          dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()
    return dataset, model


def infer_frame(dataset, model, spoof_points5, frame_id, token):
    """Forward pass on an attacked cloud; returns a prediction anno dict."""
    input_dict = {
        "points": spoof_points5,
        "frame_id": frame_id,
        "metadata": {"token": token},
    }
    data_dict = dataset.prepare_data(data_dict=input_dict)  # no gt_boxes -> pure inference
    batch = dataset.collate_batch([data_dict])
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred_dicts, _ = model(batch)
    annos = dataset.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)
    return annos[0]


def score_frame(attack_type, frame_pred, spoof_gt, iou_thresh):
    """
    Returns (is_success, matched_score, matched_label, max_iou).
    inject : success = a prediction overlaps the phantom (IoU >= thresh).
    removal: success = NO prediction overlaps the erased box (IoU < thresh).
    """
    pred_boxes = frame_pred["boxes_lidar"]
    if pred_boxes.shape[0] == 0:
        # no predictions at all
        return (attack_type == "removal"), None, None, 0.0

    pred = torch.tensor(pred_boxes[:, :7]).cuda().float()
    gt = torch.tensor(np.asarray(spoof_gt[:7], dtype=np.float32)).unsqueeze(0).cuda().float()
    iou = iou3d_nms_utils.boxes_iou3d_gpu(pred, gt)
    max_iou = iou.max().item()
    idx = int(torch.argmax(iou).item())
    score = float(frame_pred["score"][idx])
    label = int(frame_pred["pred_labels"][idx])

    if attack_type == "inject":
        return (max_iou >= iou_thresh), score, label, max_iou
    else:  # removal
        return (max_iou < iou_thresh), score, label, max_iou


def run(args, logger):
    dataset, model = build_model_and_data(args, logger)
    from nuscenes.nuscenes import NuScenes
    # NuScenesDataset sets root_path = DATA_PATH / VERSION, and the devkit wants
    # a dataroot whose /<version> subdir holds the JSON tables -> use root_path.
    nusc = NuScenes(version=args.version, dataroot=str(dataset.root_path),
                    verbose=False)
    scene_to_idx, idx_to_scene, _ = C.build_scene_index(dataset, nusc)

    os.makedirs(args.pred_dir, exist_ok=True)

    # aggregate counters
    n_targets = 0
    n_success = 0
    n_zero_pred = 0
    matched_scores = []
    label_hist = {}
    per_scene = {}

    for scene, indices in scene_to_idx.items():
        seg_file = os.path.join(args.dataset, f"{scene}_d.pkl")
        if not os.path.exists(seg_file):
            logger.info("no split file for %s, skipping", scene)
            continue
        with open(seg_file, "rb") as f:
            seg_dataset = pkl.load(f)

        preds = []
        s_targets = s_success = 0
        for local_idx, gidx in enumerate(indices):
            frame = seg_dataset[local_idx]
            # frame ordering must match the generator's (same build_scene_index)
            assert frame["frame_id"] == Path(dataset.infos[gidx]["lidar_path"]).stem, (
                f"frame misalignment in {scene} at {local_idx}")

            spoof_gt = frame.get("spoof_gt")
            # ASR is only defined on target frames; skip inference elsewhere
            # unless the caller wants a full prediction dump (--infer_all).
            if spoof_gt is None and not args.infer_all:
                continue

            spoof_pts5 = frame["points"][:, :-1]  # drop lag col -> (N,5)
            pred = infer_frame(dataset, model, spoof_pts5,
                               frame["frame_id"], frame["metadata"]["token"])
            if args.infer_all:
                preds.append(pred)

            if spoof_gt is None:
                continue
            n_targets += 1
            s_targets += 1
            if pred["boxes_lidar"].shape[0] == 0:
                n_zero_pred += 1

            ok, score, label, max_iou = score_frame(
                args.attack_type, pred, spoof_gt, args.iou_thresh)
            if ok:
                n_success += 1
                s_success += 1
                if score is not None:
                    matched_scores.append(score)
                if label is not None:
                    label_hist[label] = label_hist.get(label, 0) + 1

        if args.infer_all:
            with open(os.path.join(args.pred_dir, f"{scene}_p.pkl"), "wb") as f:
                pkl.dump(preds, f)
        per_scene[scene] = {"targets": s_targets, "success": s_success}
        logger.info("scene %s: %d/%d target frames attacked-success",
                    scene, s_success, s_targets)

    asr = (n_success / n_targets) if n_targets else 0.0
    names = np.array(cfg.CLASS_NAMES)
    label_names = {names[l - 1]: c for l, c in sorted(label_hist.items())}

    summary = {
        "attack_type": args.attack_type,
        "dataset": args.dataset,
        "iou_thresh": args.iou_thresh,
        "num_target_frames": n_targets,
        "num_success": n_success,
        "asr": asr,
        "zero_pred_frames": n_zero_pred,
        "matched_scores": np.array(matched_scores),
        "mean_matched_score": float(np.mean(matched_scores)) if matched_scores else 0.0,
        "success_label_hist": label_names,
        "per_scene": per_scene,
    }
    with open(args.asr_path, "wb") as f:
        pkl.dump(summary, f)

    logger.info("===== ASR SUMMARY (%s) =====", args.attack_type)
    logger.info("ASR = %.3f  (%d/%d target frames)", asr, n_success, n_targets)
    logger.info("zero-prediction frames: %d", n_zero_pred)
    logger.info("mean matched score: %.3f", summary["mean_matched_score"])
    logger.info("success detected-as: %s", json.dumps(label_names))
    logger.info("saved ASR summary -> %s", args.asr_path)
    return summary


def main():
    args = parse_args()
    logger = setup_logger(args.log_file)
    try:
        run(args, logger)
    except Exception:
        logger.error("===== ASR EVAL FAILED =====")
        logger.error(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
