"""Seed MSF's CP4F proposal cache from OpenPCDet CP4F predictions (official MSF protocol).

MSF's official eval reads its RPN boxes from OpenPCDet CP4F output (cfg USE_PREDBOX /
ROI_BOXES_PATH = OpenPCDet result_fixed.pkl). The old ATLAS MSF eval instead ran CP4F inside
the MSF process with the MSF pcdet fork, which gives ~1.37x more boxes and different scores on
clean frames than OpenPCDet. This writes, per segment, {local_idx: {pred_boxes (N, 9, raw
velocity), pred_scores, pred_labels}} from <run>/evaluation/cp4f_atlas/<label>/predictions
(the verified OpenPCDet CP4F eval), checks frame_id alignment with the dataset, and checks the
seeded boxes against OpenPCDet's clean full-val output on windows with no attacked frame (same
gate as the PTT eval: box count within 5%, median top-20 score gap <= 0.01).

    python seed_msf_proposals.py <run_dir> <out_dir> --label gt_epoch36 --fullval <result.pkl>
"""
import argparse
import io
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')


def arr(x, dt):
    return (x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)).astype(dt)


ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('out')
ap.add_argument('--label', default='gt_epoch36')
ap.add_argument('--fullval', required=True)
a = ap.parse_args()
run, out = Path(a.run), Path(a.out)
pred_dir = run / 'evaluation' / 'cp4f_atlas' / a.label / 'predictions'
segs = sorted(p.stem for p in (run / 'dataset').glob('*.pkl'))
missing = [s for s in segs if not (pred_dir / f'{s}_p.pkl').is_file()]
if missing:
    sys.exit(f'ERROR: {len(missing)}/{len(segs)} CP4F prediction files missing in {pred_dir}')
out.mkdir(parents=True, exist_ok=True)
fv = {str(d['frame_id']): d for d in pickle.load(open(a.fullval, 'rb'))}
ratios, gaps = [], []
for s in segs:
    frames = pickle.load(open(run / 'dataset' / f'{s}.pkl', 'rb'))
    preds = pickle.load(open(pred_dir / f'{s}_p.pkl', 'rb'))
    if len(preds) != len(frames):
        sys.exit(f'ERROR: {s}: {len(preds)} predictions vs {len(frames)} frames')
    att = [f.get('spoof_gt') is not None for f in frames]
    cache = {}
    for li, (fr, p) in enumerate(zip(frames, preds)):
        if str(p['frame_id']) != str(fr['frame_id']):
            sys.exit(f'ERROR: {s}[{li}]: prediction frame {p["frame_id"]} != dataset frame {fr["frame_id"]}')
        boxes = arr(p['boxes_lidar'], np.float32)
        if boxes.ndim != 2 or (len(boxes) and boxes.shape[1] != 9):
            sys.exit(f'ERROR: {s}[{li}]: boxes_lidar shape {boxes.shape}, need (N, 9) with velocity')
        scores = arr(p['score'], np.float32)
        cache[li] = {'pred_boxes': boxes.reshape(-1, 9), 'pred_scores': scores,
                     'pred_labels': arr(p['pred_labels'], np.int64)}
        ref = fv.get(str(fr['frame_id']))
        if ref is not None and not any(att[max(li - k, 0)] for k in range(4)):
            b = arr(ref['score'], np.float32)
            ratios.append(len(scores) / max(len(b), 1))
            A, B = np.sort(scores)[::-1], np.sort(b)[::-1]
            m = min(20, len(A), len(B))
            if m:
                gaps.append(float(np.abs(A[:m] - B[:m]).mean()))
    tmp = out / f'{s}_cp4f.pkl.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(cache, f)
    os.replace(tmp, out / f'{s}_cp4f.pkl')
r, g = float(np.median(ratios)), float(np.median(gaps))
print(f'seeded {len(segs)} segments from {pred_dir}')
print(f'clean-window check vs OpenPCDet full-val: {len(ratios)} frames, box-count ratio median {r:.4f} '
      f'[p05 {np.percentile(ratios, 5):.3f}, p95 {np.percentile(ratios, 95):.3f}], top-20 score gap median {g:.5f}')
if not (abs(r - 1) <= 0.05 and g <= 0.01):
    sys.exit('ERROR: seeded proposals do not match OpenPCDet full-val output')
print('PROPOSAL CHECK OK')
