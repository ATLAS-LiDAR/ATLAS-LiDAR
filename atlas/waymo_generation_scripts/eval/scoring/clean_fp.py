"""Clean false-positive check, GT sampling vs no GT sampling (CPU).

    python clean_fp.py [--out <GEN_ROOT>/results/_tables_all]

Hypothesis tested: GT sampling makes detectors generally more permissive (more Vehicle boxes on
nothing) without raising recall. Data: the clean frames of the 50 GT-vs-no-GT segments, predictions
saved by the clean-recall evals (PTT/neurips/generation_runs/clean_recall/<det>/<label>/predictions.pkl).
A Vehicle prediction is a clean FP if its BEV IoU with every annotated box (any class, any point
count) is < 0.1. Reported per checkpoint: FPs per frame at score >= 0.1/0.3/0.5/0.7 and, at 0.1,
by range of the box (0-15/15-25/25-40/40-55/55-80 m).
"""
import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

CR = Path('/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/PTT/neurips/generation_runs/clean_recall')
THRS = (0.1, 0.3, 0.5, 0.7)
BINS = (0, 15, 25, 40, 55, 80)


def poly(b):
    x, y, _, l, w, _, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return Polygon(p @ np.array([[c, s], [-s, c]]) + [x, y])


def run(label_dir):
    d = pickle.load(open(label_dir / 'predictions.pkl', 'rb'))
    n_fr = len(d['gts'])
    fp = np.zeros(len(THRS)); fp_r = np.zeros(len(BINS) - 1); tot = np.zeros(len(THRS))
    for pred, gt in zip(d['det_annos'], d['gts']):
        pb = np.asarray(pred['boxes_lidar'], float); pl = np.asarray(pred['pred_labels']); ps = np.asarray(pred['score'])
        keep = (pl == 1) & (ps >= THRS[0])
        pb, ps = pb[keep], ps[keep]
        if not len(pb):
            continue
        g = np.asarray(gt['boxes'], float)[:, :7] if len(gt['boxes']) else np.zeros((0, 7))
        gp = [poly(b) for b in g]
        for b, s in zip(pb, ps):
            near = np.flatnonzero(np.hypot(g[:, 0] - b[0], g[:, 1] - b[1]) <= np.hypot(g[:, 3], g[:, 4]) / 2 + np.hypot(b[3], b[4]) / 2) if len(g) else []
            pbp = poly(b)
            best = 0.0
            for j in near:
                inter = pbp.intersection(gp[j]).area
                if inter > 0:
                    best = max(best, inter / (pbp.area + gp[j].area - inter))
            is_fp = best < 0.1
            for k, t in enumerate(THRS):
                if s >= t:
                    tot[k] += 1; fp[k] += is_fp
            if is_fp:
                r = np.hypot(b[0], b[1])
                k = np.searchsorted(BINS, r, side='right') - 1
                if 0 <= k < len(fp_r):
                    fp_r[k] += 1
    return label_dir.parent.name, label_dir.name, n_fr, (fp / n_fr).tolist(), (fp / np.maximum(tot, 1)).tolist(), (fp_r / n_fr).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/storage/scratch1/9/spanse30/cvpr_datasets/results/_tables_all')
    ap.add_argument('--workers', type=int, default=28)
    a = ap.parse_args()
    dirs = sorted(p.parent for p in CR.glob('*/*/predictions.pkl'))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, dirs))
    R = {(d, l): (n, f, sh, fr) for d, l, n, f, sh, fr in res}
    rec = {(d, l): json.load(open(CR / d / l / 'recall.json')).get('recall_vehicle_iou0.7') for d, l in R}
    md = ['# Clean false positives: GT sampling vs no GT sampling (clean frames of the 50 GT-vs-no-GT segments)', '',
          'Clean FP = Vehicle prediction with BEV IoU < 0.1 to every annotated box. FP/frame at score >= 0.1/0.3/0.5/0.7; '
          'share = FP / all Vehicle predictions at that cutoff; range bins at 0.1 (FP/frame).', '',
          '| Detector | Checkpoint | recall@0.7 | FP/frame 0.1/0.3/0.5/0.7 | FP share 0.1/0.3/0.5/0.7 | FP/frame by range 0-15/15-25/25-40/40-55/55-80 m |',
          '|---|---|---|---|---|---|']
    ep = lambda l: int(l.split('epoch')[1])  # noqa: E731
    for d in sorted({k[0] for k in R}):
        for l in sorted((k[1] for k in R if k[0] == d), key=lambda l: (ep(l), l)):
            n, f, sh, fr = R[(d, l)]
            r = rec[(d, l)]
            md.append(f'| {d} | {l} | {100 * r:.1f} | ' if r is not None else f'| {d} | {l} | — | ')
            md[-1] += '/'.join(f'{x:.2f}' for x in f) + ' | ' + '/'.join(f'{100 * x:.1f}%' for x in sh) + ' | ' + '/'.join(f'{x:.2f}' for x in fr) + ' |'
    md += ['', '## Final epoch, GT − no-GT (FP/frame at 0.1 · 0.7)', '']
    for d, e in (('cp_sf', 30), ('cp4f', 36), ('ptt32', 6)):
        g, n = R.get((d, f'gt_epoch{e}')), R.get((d, f'nogt_epoch{e}'))
        if g and n:
            md.append(f'- {d} e{e}: GT {g[1][0]:.2f} · {g[1][3]:.2f} vs no-GT {n[1][0]:.2f} · {n[1][3]:.2f}  '
                      f'(Δ {g[1][0] - n[1][0]:+.2f} · {g[1][3] - n[1][3]:+.2f})')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'clean_fp.md').write_text('\n'.join(md) + '\n')
    json.dump({f'{d}/{l}': {'frames': v[0], 'fp_per_frame': v[1], 'fp_share': v[2], 'fp_per_frame_by_range': v[3]} for (d, l), v in R.items()},
              open(out / 'clean_fp.json', 'w'), indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
