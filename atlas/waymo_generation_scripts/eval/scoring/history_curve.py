"""Attack lifetime (k-history) curves + score heatmaps from saved predictions. CPU only, no forward pass.

    python history_curve.py <run> [<run> ...] [--dets cp4f msf4 msf8 ptt cpsf] [--workers 16]

Reproduces the paper's Fig. attack_lifetime_injection (ASR vs attack horizon) and Fig.
injectionheatmap (score density of successful phantoms vs horizon) for CP4F (W=4), MSF-4 (W=4),
MSF-8 (W=8), PTT-32 (W=32); CP-SF (W=1) is available as a baseline. Definitions follow the
original MSF/eval_history_{curve,decay}_*.py, with the attack windows taken from annos
(window_start_local..window_end_local) instead of the old per-detector timing arrays:
  onset   horizon h = min(p, W) for the p-th frame of an attack window (p = 1, 2, ...): h < W are
          the first frames of the block, h = W pools every later block frame (history saturated);
  decay   horizon h = W + d for the d-th clean frame after the window (d = 1..W, stopping at the
          next window or the segment end); the phantom box is the last spoofed frame's spoof_gt
          carried in the WORLD frame (as the old decay scripts did, for global and relative runs)
          and, for relative runs, also the EGO-fixed box (reported as decay_ego);
  h = 0   is clean history (plotted as 0, as in the paper).
Hit rule = atlas_metrics.best_match + hits (argmax-IoU of any class, target class at its IoU
threshold, score >= t for t in 0.1/0.3/0.5/0.7), the same rule as the main ASR table; onset and
decay frames without a spoof box are skipped. Writes <run>/evaluation/history/<det>.json with ASR
per horizon and threshold, per-segment counts (segment-bootstrap CIs), and per-horizon score
histograms (10 bins on [0, 1]) of the t = 0.1 hits.
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_metrics import ROOT, THRS, best_match, hits, spoof_cache  # noqa: E402

DETS = {'cpsf': ('cp_sf_atlas/gt_epoch30', 1), 'cp4f': ('cp4f_atlas/gt_epoch36', 4),
        'msf4': ('msf4_cvpr', 4), 'msf8': ('msf8_cvpr', 8), 'ptt': ('ptt32_atlas/gt_epoch6', 32)}
BINS = np.linspace(0, 1, 11)


def yaw_of(pose):
    return np.arctan2(pose[1, 0], pose[0, 0])


def ego_to_world(box, pose):
    b = box.copy()
    b[:3] = (pose @ np.r_[box[:3], 1.0])[:3]
    b[6] = box[6] + yaw_of(pose)
    return b


def world_to_ego(box, pose):
    b = box.copy()
    b[:3] = (np.linalg.inv(pose) @ np.r_[box[:3], 1.0])[:3]
    b[6] = box[6] - yaw_of(pose)
    return b


def windows(run, seg):
    a = pickle.load(open(ROOT / run / 'annos' / f'{seg}_a.pkl', 'rb'))
    return sorted((int(x['window_start_local']), int(x['window_end_local'])) for x in a['spoof_annos'])


def _segment(args):
    run, seg, pred_file, spoof, W, relative = args
    preds = pickle.load(open(pred_file, 'rb'))
    wins = windows(run, seg)
    n = len(spoof)
    in_win = np.zeros(n, bool)
    for s, e in wins:
        in_win[s:e + 1] = True
    out = {}                                     # (kind, h) -> list of (target-hit array, score@0.1)

    def add(kind, h, box, i):
        iou, lab, sc, _ = best_match(preds[i], box)
        t_hit, _ = hits({'iou': iou, 'label': lab}, spoof[i].get('cls', 1) if spoof[i]['gt'] is not None else 1)
        out.setdefault((kind, h), []).append((t_hit.astype(int), float(sc[0])))

    for s, e in wins:
        last = None
        for i in range(s, min(e, n - 1) + 1):
            g = spoof[i]['gt']
            if g is None:
                continue
            last = i
            add('onset', min(i - s + 1, W), g, i)
        if last is None:
            continue
        g_world = ego_to_world(spoof[last]['gt'], spoof[last]['pose'])
        for d in range(1, W + 1):
            i = e + d
            if i >= n or in_win[i]:
                break
            add('decay', W + d, world_to_ego(g_world, spoof[i]['pose']), i)
            if relative:
                add('decay_ego', W + d, spoof[last]['gt'], i)
    return seg, out


def run_history(run, det, workers):
    sub, W = DETS[det]
    d = ROOT / run / 'evaluation' / sub
    if not (d / 'summary.pkl').is_file() and not (d / 'asr_only.json').is_file():
        print(f'{run} {det}: no finished eval at {d}, skipped')
        return None
    spoof = spoof_cache(run, workers)
    relative = 'rel_' in run
    jobs = [(run, s, d / 'predictions' / f'{s}_p.pkl', spoof[s], W, relative) for s in sorted(spoof)]
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(_segment, jobs))
    T = len(THRS)
    curves = {}
    for kind in ('onset', 'decay', 'decay_ego'):
        hs = sorted({h for r in res.values() for (k, h) in r if k == kind})
        for h in hs:
            per_seg = {}
            scores = []
            for s, r in res.items():
                rows = r.get((kind, h), [])
                if rows:
                    hm = np.array([x[0] for x in rows])
                    per_seg[s] = [len(rows)] + hm.sum(0).tolist()
                    scores += [sc for hit, sc in rows if hit[0]]
            A = np.array(list(per_seg.values()), float)
            asr = A[:, 1:].sum(0) / A[:, 0].sum()
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(A), (2000, len(A)))
            bs = A[idx][:, :, 1:].sum(1) / A[idx][:, :, 0].sum(1, keepdims=True)
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            curves.setdefault(kind, {})[str(h)] = {
                'n_frames': int(A[:, 0].sum()), 'asr': dict(zip(map(str, THRS), map(float, asr))),
                'ci95': {str(t): [float(lo[k]), float(hi[k])] for k, t in enumerate(THRS)},
                'score_hist_0.1': np.histogram(scores, BINS)[0].tolist(),
                'per_segment': per_seg}
    out = {'run': run, 'det': det, 'eval': sub, 'window': W, 'thresholds': list(THRS),
           'bins': BINS.tolist(), 'curves': curves}
    p = ROOT / run / 'evaluation' / 'history' / f'{det}.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, 'w'))
    on = curves.get('onset', {})
    de = curves.get('decay', {})
    print(f'{run:28s} {det:5s} W={W:2d} onset ' + ' '.join(f'{100 * on[h]["asr"]["0.1"]:4.1f}' for h in sorted(on, key=int))
          + ' | decay ' + ' '.join(f'{100 * de[h]["asr"]["0.1"]:4.1f}' for h in sorted(de, key=int)[:8]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--dets', nargs='+', default=['cp4f', 'msf4', 'msf8', 'ptt'], choices=list(DETS))
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    for run in a.runs:
        for det in a.dets:
            run_history(run, det, a.workers)


if __name__ == '__main__':
    main()
