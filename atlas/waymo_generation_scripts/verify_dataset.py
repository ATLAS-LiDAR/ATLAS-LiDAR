"""Post-generation check for one regenerated ATLAS dataset dir (<root>/{dataset,annos}).

Fails (exit 1) unless: segment and annos counts match --expect; every frame is a numpy
float32 (N, 6) cloud; every spoof intensity is <= 1 (tanh; float32 tanh of large raw values is exactly 1.0, as for real points
by pcdet). Checks --sample segments (all if 0) to keep it fast.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--expect', type=int, required=True)
ap.add_argument('--sample', type=int, default=3)
a = ap.parse_args()
root = Path(a.root)
segs = sorted((root / 'dataset').glob('*.pkl'))
annos = sorted((root / 'annos').glob('*_a.pkl'))
bad = []
if len(segs) != a.expect or len(annos) != a.expect:
    bad.append(f'counts: {len(segs)} dataset / {len(annos)} annos, expected {a.expect}')
pick = segs if a.sample <= 0 else [segs[i] for i in np.linspace(0, len(segs) - 1, min(a.sample, len(segs))).astype(int)]
n_att = n_sp = n_hi = 0
imax = 0.0
for s in pick:
    for fr in pickle.load(open(s, 'rb')):
        p = fr['points']
        if not isinstance(p, np.ndarray) or p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 6:
            bad.append(f'{s.name}: points {type(p).__name__} {getattr(p, "dtype", None)} {getattr(p, "shape", None)}')
            break
        if fr.get('spoof_gt') is None:
            continue
        s0, e0 = int(fr['final_spoof_start_index']), int(fr['final_spoof_end_index'])
        n_att += 1
        if e0 > s0:
            I = p[s0:e0, 3]
            n_sp += len(I); n_hi += int((I > 1.0).sum()); imax = max(imax, float(I.max()))
if n_hi:
    bad.append(f'{n_hi}/{n_sp} spoof points with intensity > 1 (max {imax:.2f})')
print(f'{root.name}: {len(segs)} segments, checked {len(pick)}: {n_att} attacked frames, {n_sp} spoof points, '
      f'max spoof intensity {imax:.3f}')
if bad:
    print('VERIFY FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1)
print('VERIFY OK')
