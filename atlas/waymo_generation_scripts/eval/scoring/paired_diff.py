"""Difference of a paper metric between two detectors / recipes on the SAME dataset, with a JOINT paired bootstrap
(the same resampled segments for both, as the paper's Uncertainty paragraph requires). CPU.

    python paired_diff.py A_paper_metrics_instances.pkl B_paper_metrics_instances.pkl [--key fired|argmax|tau_f]

The instance files are written by paper_metrics.py next to each paper_metrics.json. Instances are aligned on
(segment, frame); only frames present in both are used. Prints metric(A), metric(B), A - B and its 95% CI.
EPR for injection sets, conditional HR for removal-type sets.
"""
import argparse
import pickle

import numpy as np


def value(kind, m, mt):
    if kind == 'injection':
        return float(np.mean(mt.astype(int) - m.astype(int)))
    return float((m & ~mt).sum() / max(m.sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--key', default='fired|argmax|tau_f')
    ap.add_argument('--reps', type=int, default=1000)
    a = ap.parse_args()
    A, B = pickle.load(open(a.a, 'rb')), pickle.load(open(a.b, 'rb'))
    if A['kind'] != B['kind']:
        raise SystemExit('different dataset kinds')
    pa, pb = A['per_instance'][a.key], B['per_instance'][a.key]
    ka = {(s, int(i)): k for k, (s, i) in enumerate(zip(pa['seg'], pa['i']))}
    pairs = [(ka[(s, int(i))], k) for k, (s, i) in enumerate(zip(pb['seg'], pb['i'])) if (s, int(i)) in ka]
    ia, ib = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    seg = pa['seg'][ia]
    ma, mta, mb, mtb = pa['m'][ia], pa['mt'][ia], pb['m'][ib], pb['mt'][ib]
    va, vb = value(A['kind'], ma, mta), value(A['kind'], mb, mtb)
    segs = np.unique(seg); by = {s: np.flatnonzero(seg == s) for s in segs}
    rng = np.random.default_rng(0); d = []
    for _ in range(a.reps):
        k = np.concatenate([by[s] for s in rng.choice(segs, len(segs))])
        d.append(value(A['kind'], ma[k], mta[k]) - value(A['kind'], mb[k], mtb[k]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    name = 'EPR' if A['kind'] == 'injection' else 'HR'
    print(f'{name}  A {va:.4f}  B {vb:.4f}  A-B {va - vb:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  '
          f'({len(pairs)} paired instances, {len(segs)} segments, key {a.key})')


if __name__ == '__main__':
    main()
