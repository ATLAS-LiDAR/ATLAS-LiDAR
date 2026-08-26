#!/usr/bin/env python
"""Generate KITTI ATLAS adversarial variants (single-frame).

Modes:
  inject  --trace_file trace_{easy,medium,hard}.pkl : place a phantom car per
           frame; writes frames where the phantom survives (>=1 point kept).
  removal --az_width {10..60} : aim the A-HFR wedge at a real car and delete
           points; writes frames where a car target was found.

Only the val split is attacked (KittiDataset with DATA_SPLIT.test = val).
Output: one <frame_id>.pkl per target frame (see kitti_atlas_common.save_frame).

Example:
  python generate_data_kitti.py --cfg_file ../tools/cfgs/dataset_configs/kitti_dataset_gen.yaml \
     --mode inject --trace_file assets_kitti/trace_easy.pkl \
     --save_dir <out>/inject_easy --log_file inj.log --seed 42
"""
import argparse
import importlib
import logging
import os
import sys
import traceback

import numpy as np

import removal
import kitti_atlas_common as K
importlib.reload(removal)

TARGET_CLASS = "Car"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", required=True)
    p.add_argument("--mode", required=True, choices=["inject", "removal"])
    p.add_argument("--trace_file", default=None, help="required for --mode inject")
    p.add_argument("--az_width", type=int, default=20, choices=[10, 20, 30, 40, 50, 60])
    p.add_argument("--save_dir", required=True)
    p.add_argument("--log_file", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


PRR = {10: 0.97, 20: 1.00, 30: 0.99, 40: 0.98, 50: 0.90, 60: 0.90}


def build_kitti(cfg_file, logger):
    import yaml
    from easydict import EasyDict
    from pathlib import Path
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    cfg = EasyDict(yaml.safe_load(open(cfg_file)))
    root = Path(__file__).resolve().parent.parent / "data" / "kitti"
    dataset = KittiDataset(dataset_cfg=cfg, class_names=cfg.get("CLASS_NAMES", ["Car", "Pedestrian", "Cyclist"]),
                           root_path=root, training=False, logger=logger)
    return dataset


def main():
    args = parse_args()
    log = logging.getLogger("kitti_atlas")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(args.log_file); fh.setFormatter(fmt); log.addHandler(fh)
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        dataset = build_kitti(args.cfg_file, log)
        infos = dataset.kitti_infos
        log.info("val frames: %d", len(infos))

        traces = None
        if args.mode == "inject":
            assert args.trace_file, "--trace_file required for inject"
            import pickle as pkl
            traces = pkl.load(open(args.trace_file, "rb"))
        p_remove = PRR[args.az_width]

        done = {f[:-4] for f in os.listdir(args.save_dir) if f.endswith(".pkl")}
        n_written = 0

        for idx, info in enumerate(infos):
            fid = info["point_cloud"]["lidar_idx"]
            if fid in done:
                n_written += 1
                continue
            boxes, names = K.frame_car_boxes(info)
            car_mask = names == TARGET_CLASS
            if not car_mask.any() and args.mode == "removal":
                continue  # no car to erase

            points, _ = K.load_points_kitti(dataset, idx)
            rng = np.random.default_rng(int(args.seed + idx))

            if args.mode == "inject":
                spoof, anno, box = K.place_trace(traces, int(args.seed + idx))
                fused, gt, n_r, n_k = K.spoof_frame(spoof, points, box, rng)
                if n_k < 1:
                    continue  # phantom fully occluded -> not a target
                anno.update({"n_spoof_r": n_r, "n_spoof_k": n_k})
                K.save_frame(args.save_dir, fid, fused, gt, anno)
            else:  # removal — aim wedge at nearest front car (>10m)
                cars = boxes[car_mask]
                cars = cars[cars[:, 0] > 10.0]  # in front, beyond min range
                if len(cars) == 0:
                    continue
                target = cars[np.argmin(np.linalg.norm(cars[:, :2], axis=1))]
                attacked, _, meta = removal.spoof_frame_ahfr_roadside(
                    points=points, target_box=target, state=None,
                    p_remove=p_remove, az_width_deg=args.az_width, rng=rng)
                meta.update({"az_width_deg": args.az_width, "p_remove": p_remove})
                K.save_frame(args.save_dir, fid, attacked.astype(np.float32), target, meta)

            n_written += 1
            if n_written % 200 == 0:
                log.info("written %d frames", n_written)

        log.info("DONE %s: %d frames written to %s", args.mode, n_written, args.save_dir)

    except Exception:
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
