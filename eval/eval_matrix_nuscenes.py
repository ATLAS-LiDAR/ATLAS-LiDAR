#!/usr/bin/env python
"""Evaluate every nuScenes model on the clean val split (mAP/NDS) and on all 12
ATLAS adversarial splits (ASR), aggregating everything into one CSV.

For each (model, checkpoint):
  * clean : standard nuScenes detection eval on v1.0-trainval val -> mAP, NDS
  * inject_{global,relative}_{easy,medium,hard} : ASR (phantom detected)
  * removal_az{10,20,30,40,50,60}               : ASR (tracked car erased)

CSV columns:
  model, config, split, metric_type, mAP, NDS, ASR, num_targets, num_success, notes

Run from tools/. Camera+LiDAR models (BEVFusion) can't run on the LiDAR-only
download -> recorded as a row with a note, not silently dropped.
"""
import argparse
import csv
import os
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

INJECT_SPLITS = [f"inject_{m}_{d}" for m in ("global", "relative")
                 for d in ("easy", "medium", "hard")]
REMOVAL_SPLITS = [f"removal_az{a}" for a in (10, 20, 30, 40, 50, 60)]

# model_name -> (config relative to tools/, checkpoint filename, needs_camera)
MODELS = [
    ("pp_multihead",              "cfgs/nuscenes_models/cbgs_pp_multihead.yaml",              "pp_multihead_nds5823_updated.pth", False),
    ("second_multihead",          "cfgs/nuscenes_models/cbgs_second_multihead.yaml",          "cbgs_second_multihead.pth",        False),
    ("dyn_pp_centerpoint",        "cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml",        "cbgs_dyn_pp_centerpoint.pth",      False),
    ("voxel01_centerpoint",       "cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml", "cbgs_voxel01_res3d_centerpoint.pth", False),
    ("voxel0075_centerpoint",     "cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml", "cbgs_voxel0075_res3d_centerpoint.pth", False),
    ("voxel0075_voxelnext",       "cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml",       "cbgs_voxel0075_voxelnext.pth",     False),
    ("transfusion_lidar",         "cfgs/nuscenes_models/transfusion_lidar.yaml",              "transfusion_lidar.pth",            False),
    ("bevfusion",                 "cfgs/nuscenes_models/bevfusion.yaml",                      "bevfusion.pth",                    True),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas_dir", required=True, help="dir with the 12 split subdirs")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--data_path", default="../data/nuscenes")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--work_dir", required=True, help="scratch dir for eval outputs")
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--only_model", default=None, help="restrict to one model name")
    p.add_argument("--skip_clean", action="store_true")
    p.add_argument("--split_suffix", default="",
                   help="suffix appended to every split name, e.g. _cont or "
                        "_intermit for the single-attack schedule variants")
    p.add_argument("--per_scene_dir", default=None,
                   help="if set, dump {split}_per_scene.pkl here (needed for "
                        "sequence-level and macro-averaged ASR)")
    return p.parse_args()


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
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=4, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()
    return dataset, loader, model


def eval_clean(dataset, loader, model, work_dir, logger):
    """Standard nuScenes detection eval -> (mAP, NDS)."""
    det_annos = []
    for batch in loader:
        load_data_to_gpu(batch)
        with torch.no_grad():
            pred_dicts, _ = model(batch)
        det_annos += dataset.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)
    result_str, result_dict = dataset.evaluation(
        det_annos, cfg.CLASS_NAMES, output_path=work_dir)
    logger.info("\n%s", result_str)
    return float(result_dict.get("mAP", float("nan"))), float(result_dict.get("NDS", float("nan")))


def infer_frame(dataset, model, spoof_points5, frame_id, token):
    data_dict = dataset.prepare_data(data_dict={
        "points": spoof_points5, "frame_id": frame_id, "metadata": {"token": token}})
    batch = dataset.collate_batch([data_dict])
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred_dicts, _ = model(batch)
    return dataset.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)[0]


def eval_asr(dataset, model, split_dir, attack_type, scene_to_idx, iou_thresh,
             per_scene_out=None):
    """Frame-pooled ASR, plus per-scene counts.

    `per_scene_out` (a dict) receives {scene: {targets, success, scheduled}} so
    downstream analysis can compute sequence-level and macro-averaged ASR. With
    the single-attack schedules the attack length varies per sequence, so a
    frame-pooled ratio weights long attacks far more heavily than short ones and
    is not on its own a fair summary.
    """
    n_targets = n_success = 0
    for scene, indices in scene_to_idx.items():
        seg_file = os.path.join(split_dir, f"{scene}_d.pkl")
        if not os.path.exists(seg_file):
            continue
        import pickle as pkl
        with open(seg_file, "rb") as f:
            seg = pkl.load(f)
        s_t = s_s = s_sched = 0
        for local_idx, gidx in enumerate(indices):
            frame = seg[local_idx]
            if frame.get("scheduled_attack"):
                s_sched += 1
            spoof_gt = frame.get("spoof_gt")
            if spoof_gt is None:
                continue
            n_targets += 1
            s_t += 1
            pred = infer_frame(dataset, model, frame["points"][:, :-1],
                               frame["frame_id"], frame["metadata"]["token"])
            pb = pred["boxes_lidar"]
            if pb.shape[0] == 0:
                if attack_type == "removal":
                    n_success += 1
                    s_s += 1
                continue
            gt = torch.tensor(np.asarray(spoof_gt[:7], dtype=np.float32)).unsqueeze(0).cuda().float()
            iou = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(pb[:, :7]).cuda().float(), gt)
            max_iou = iou.max().item()
            hit = max_iou >= iou_thresh
            if (attack_type == "inject" and hit) or (attack_type == "removal" and not hit):
                n_success += 1
                s_s += 1
        if per_scene_out is not None:
            per_scene_out[scene] = {"targets": s_t, "success": s_s,
                                    "scheduled": s_sched}
    asr = (n_success / n_targets) if n_targets else float("nan")
    return asr, n_targets, n_success


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    logger = common_utils.create_logger(os.path.join(args.work_dir, "eval_matrix.log"))

    rows = []
    write_header = not os.path.exists(args.out_csv)
    csv_f = open(args.out_csv, "a", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(["model", "config", "split", "metric_type",
                         "mAP", "NDS", "ASR", "num_targets", "num_success", "notes"])
        csv_f.flush()

    for name, cfg_file, ckpt_name, needs_cam in MODELS:
        if args.only_model and name != args.only_model:
            continue
        ckpt = os.path.join(args.ckpt_dir, ckpt_name)
        if needs_cam:
            writer.writerow([name, cfg_file, "clean", "skipped", "", "", "", "", "",
                             "camera+LiDAR model; LiDAR-only download cannot run it"])
            csv_f.flush()
            logger.info("SKIP %s (needs camera)", name)
            continue
        if not os.path.exists(ckpt):
            writer.writerow([name, cfg_file, "clean", "error", "", "", "", "", "",
                             f"checkpoint missing: {ckpt_name}"])
            csv_f.flush()
            logger.info("MISSING ckpt %s", ckpt_name)
            continue

        try:
            logger.info("===== MODEL %s =====", name)
            dataset, loader, model = build(cfg_file, ckpt, args.version, args.data_path, logger)

            # clean mAP/NDS
            if not args.skip_clean:
                mAP, NDS = eval_clean(dataset, loader, model, args.work_dir, logger)
                writer.writerow([name, cfg_file, "clean", "detection",
                                 f"{mAP:.4f}", f"{NDS:.4f}", "", "", "", ""])
                csv_f.flush()
                logger.info("%s clean: mAP=%.4f NDS=%.4f", name, mAP, NDS)

            # ASR on all 12 splits
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version=args.version, dataroot=str(dataset.root_path), verbose=False)
            scene_to_idx, _, _ = C.build_scene_index(dataset, nusc)

            for base in INJECT_SPLITS + REMOVAL_SPLITS:
                split = f"{base}{args.split_suffix}"
                atk = "inject" if base.startswith("inject") else "removal"
                split_dir = os.path.join(args.atlas_dir, split)
                if not os.path.isdir(split_dir):
                    writer.writerow([name, cfg_file, split, "asr", "", "", "", "", "",
                                     "split dir missing"])
                    csv_f.flush()
                    continue
                per_scene = {} if args.per_scene_dir else None
                asr, nt, ns = eval_asr(dataset, model, split_dir, atk, scene_to_idx,
                                       args.iou_thresh, per_scene_out=per_scene)
                writer.writerow([name, cfg_file, split, "asr", "", "",
                                 f"{asr:.4f}", nt, ns, atk])
                csv_f.flush()
                logger.info("%s %s: ASR=%.4f (%d/%d)", name, split, asr, ns, nt)
                if per_scene is not None:
                    import pickle as pkl
                    os.makedirs(args.per_scene_dir, exist_ok=True)
                    with open(os.path.join(args.per_scene_dir,
                                           f"{name}__{split}_per_scene.pkl"), "wb") as pf:
                        pkl.dump({"model": name, "split": split,
                                  "attack_type": atk, "per_scene": per_scene}, pf)

            del model, dataset, loader
            torch.cuda.empty_cache()

        except Exception:
            writer.writerow([name, cfg_file, "", "error", "", "", "", "", "",
                             traceback.format_exc().splitlines()[-1]])
            csv_f.flush()
            logger.error("MODEL %s FAILED:\n%s", name, traceback.format_exc())

    csv_f.close()
    logger.info("wrote CSV -> %s", args.out_csv)


if __name__ == "__main__":
    main()
