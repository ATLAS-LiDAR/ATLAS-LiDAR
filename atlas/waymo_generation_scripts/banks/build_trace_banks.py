"""Build the ATLAS car trace libraries traces_{easy,medium,hard}.pkl (the spoof shapes of every car injection).

Script version of the trace-collection cell that produced the original libraries: iterate the single-frame
Waymo VAL samples in loader order (OpenPCDet waymo_models/centerpoint.yaml, test mode), and for every Vehicle
box take all points inside the box (box-aligned, no enlargement). A trace is kept by a library when its point
count is inside the library's band; each library keeps the first 300 traces in iteration order.

    python build_trace_banks.py --cfg <OpenPCDet>/tools/cfgs/waymo_models/centerpoint.yaml \
        --data_path <Waymo root> --out_dir <dir> [--verify_dir <dir with the original pickles>]

Verified against the original pickles: identical traces (frame, box, num_points) and identical point sets; the
row order inside a trace follows the loader's point order and can differ from the original. Some per-point draws
in the generators follow that order, so datasets built from a rebuilt library are statistically identical to the
paper's, not byte-identical; use the original pickles for byte-exact reproduction.

Output records: {'frame': val sample index, 'box': (8,) x,y,z,dx,dy,dz,yaw,label, 'points': (N,5) x,y,z,i,e as
loaded by pcdet (tanh-scaled intensity/elongation), 'num_points': N}.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

BANDS = {'easy': (100, 200), 'medium': (300, 500), 'hard': (500, 1000)}   # inclusive point-count bands


def points_in_box(points, box):
    cx, cy, cz, dx, dy, dz, heading = box[:7]
    shifted = points[:, :3] - np.array([cx, cy, cz])
    c, s = np.cos(-heading), np.sin(-heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = shifted @ R.T
    return (np.abs(local[:, 0]) <= dx / 2) & (np.abs(local[:, 1]) <= dy / 2) & (np.abs(local[:, 2]) <= dz / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_bank', type=int, default=300)
    ap.add_argument('--max_frames', type=int, default=10000)
    ap.add_argument('--verify_dir', default=None)
    a = ap.parse_args()

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    cfg_from_yaml_file(a.cfg, cfg)
    cfg.DATA_CONFIG.DATA_PATH = a.data_path
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
                                     dist=False, workers=0, logger=logger, training=False)

    banks = {k: [] for k in BANDS}
    for i in range(min(a.max_frames, len(dataset))):
        sample = dataset[i]
        boxes, points = sample['gt_boxes'], sample['points']
        for box in boxes[boxes[:, 7] == 1]:
            inside = points[points_in_box(points, box)]
            n = len(inside)
            if n == 0:
                continue
            for k, (lo, hi) in BANDS.items():
                if lo <= n <= hi and len(banks[k]) < a.per_bank:
                    banks[k].append({'frame': i, 'box': box, 'points': inside, 'num_points': n})
        if all(len(v) == a.per_bank for v in banks.values()):
            break

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ok = True
    for k, v in banks.items():
        pickle.dump(v, open(out / f'traces_{k}.pkl', 'wb'))
        msg = f'traces_{k}.pkl: {len(v)} traces, frames {v[0]["frame"]}-{v[-1]["frame"]}, points {min(t["num_points"] for t in v)}-{max(t["num_points"] for t in v)}'
        if a.verify_dir:
            ref = pickle.load(open(Path(a.verify_dir) / f'traces_{k}.pkl', 'rb'))
            same = len(ref) == len(v) and all(
                r['frame'] == t['frame'] and r['num_points'] == t['num_points'] and np.allclose(r['box'], t['box'])
                and np.allclose(np.sort(r['points'], 0), np.sort(t['points'], 0)) for r, t in zip(ref, v))
            ok &= same
            msg += '  == original (same traces and point sets; row order may differ)' if same else '  DIFFERS from original'
        print(msg)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
