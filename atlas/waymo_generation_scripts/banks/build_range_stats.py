"""Re-fit data/range_conditioned_stats.npy (used ONLY by the adaptive intensity mode).

Structure of the shipped file (one entry per spoof range band, metres): {(6,12), (12,18), (18,24)} ->
  log_mu, log_sigma   lognormal fit of RAW Waymo intensity (before pcdet's tanh; generators apply tanh after drawing)
  intensity_p1/p99    clip range of the draws
  elong_pool          raw elongation values sampled with replacement
  P_two               fraction of points that are second returns
  n_samples           number of points in the fit
The shipped file was fit interactively (the original notebook cell is not preserved). This script re-fits the
same quantities from points inside Vehicle boxes of Waymo val frames, banded by the box centre's planar range;
it is a documented re-fit, NOT a bit-exact reproduction of the shipped file. Keep the shipped file to reproduce
the paper; use this for a new sensor/dataset (the nuScenes/KITTI ports use the same structure).

    python build_range_stats.py --waymo_root <Waymo> --out stats.npy [--frames 200]
"""
import argparse
import pickle

import numpy as np

BANDS = [(6, 12), (12, 18), (18, 24)]


def inside(points, box):
    c, d, h = box[:3], box[3:6], box[6]
    p = points[:, :3] - c
    ca, sa = np.cos(-h), np.sin(-h)
    lx, ly = p[:, 0] * ca - p[:, 1] * sa, p[:, 0] * sa + p[:, 1] * ca
    return (np.abs(lx) <= d[0] / 2) & (np.abs(ly) <= d[1] / 2) & (np.abs(p[:, 2]) <= d[2] / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--waymo_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=200)
    a = ap.parse_args()
    W = a.waymo_root
    segs = [l.strip().replace('.tfrecord', '') for l in open(f'{W}/ImageSets/val.txt') if l.strip()]
    per = {b: {'I': [], 'E': [], 'two': []} for b in BANDS}
    n = 0
    for sg in segs:
        for info in pickle.load(open(f'{W}/waymo_processed_data_v0_5_0/{sg}/{sg}.pkl', 'rb')):
            if n >= a.frames:
                break
            P = np.load(f'{W}/waymo_processed_data_v0_5_0/{sg}/{info["point_cloud"]["sample_idx"]:04d}.npy')
            n_first = int(info['num_points_of_each_lidar'][0]) if 'num_points_of_each_lidar' in info else len(P)
            ann = info['annos']
            for box in ann['gt_boxes_lidar'][ann['name'] == 'Vehicle']:
                r = float(np.hypot(box[0], box[1]))
                for b in BANDS:
                    if b[0] <= r < b[1]:
                        m = inside(P, box)
                        per[b]['I'].append(P[m, 3]); per[b]['E'].append(P[m, 4])
                        per[b]['two'].append(np.flatnonzero(m) >= n_first)
            n += 1
        if n >= a.frames:
            break
    stats = {}
    for b, v in per.items():
        I = np.concatenate(v['I']); E = np.concatenate(v['E']); two = np.concatenate(v['two'])
        pos = I[I > 0]
        stats[b] = {'log_mu': float(np.log(pos).mean()), 'log_sigma': float(np.log(pos).std()),
                    'intensity_p1': float(np.percentile(I, 1)), 'intensity_p99': float(np.percentile(I, 99)),
                    'elong_pool': E.astype(np.float32), 'P_two': float(two.mean()), 'n_samples': int(I.size)}
        print(b, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in stats[b].items() if k != 'elong_pool'})
    np.save(a.out, stats, allow_pickle=True)


if __name__ == '__main__':
    main()
