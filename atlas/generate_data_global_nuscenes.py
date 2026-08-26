#!/usr/bin/env python
"""Global (world-fixed) point-injection adversarial split for nuScenes.

nuScenes analogue of generate_data_global.py. See nuscenes_atlas_common.py for
the shared machinery (assets, physics, scene grouping). Example:

  python generate_data_global_nuscenes.py \
      --cfg_file ../tools/cfgs/dataset_configs/nuscenes_dataset_mini.yaml \
      --save_dir_dataset <out>/global \
      --save_dir_annos   <out>/global_annos \
      --trace_file trace_medium.pkl \
      --log_file global.log --seed 42
"""
import argparse
import logging
import sys
import traceback

import nuscenes_atlas_common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", required=True, help="nuScenes dataset cfg yaml")
    p.add_argument("--save_dir_dataset", required=True)
    p.add_argument("--save_dir_annos", required=True)
    p.add_argument("--trace_file", required=True)
    p.add_argument("--log_file", required=True)
    p.add_argument("--seed", type=int, default=42)
    C.add_schedule_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    log = logging.getLogger("spoof")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(args.log_file); fh.setFormatter(fmt); log.addHandler(fh)

    try:
        C.run_injection(args, mode="global", log=log)
    except Exception:
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
