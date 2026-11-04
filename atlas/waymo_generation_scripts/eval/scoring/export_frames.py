"""Per-attacked-frame export of every scored (run, detector) for offline analysis. CPU.

    python export_frames.py [--out <GEN_ROOT>/results/frames] [--workers 16] [runs ...]

One parquet per run: <out>/<run>.parquet, one row per (detector label, attacked frame):
  det, segment, frame, run_id, win_pos (1-based frame index inside the attack window), win_len,
  range_m (planar distance ego -> spoof box centre), n_kept (spoof points after raycasting),
  n_cand (before), ego_speed (m/s, from consecutive poses at 10 Hz), cls (target class id),
  iou_t / label_t / score_t for t in 0.1/0.3/0.5/0.7 (argmax-IoU prediction among score >= t),
  hit_t (target-class hit, the ASR rule), any_t (any-class hit), n_pred (predictions in the frame).
ASR at cutoff t = mean(hit_t) over a run's rows of one det. Runs default to every folder with
evaluation/atlas_metrics.json; detectors = the keys scored there.
"""
import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_metrics import ROOT, THRS, best_match, hits  # noqa: E402

TT = [str(t).replace('.', '') for t in THRS]   # 01 03 05 07


def seg_rows(args):
    run, det, seg = args
    ds = ROOT / run / 'dataset' / f'{seg}.pkl'
    pf = ROOT / run / 'evaluation' / det / 'predictions' / f'{seg}_p.pkl'
    if not pf.is_file():
        return []
    frames = pickle.load(open(ds, 'rb'))
    preds = pickle.load(open(pf, 'rb'))
    poses = [np.asarray(f['pose'], float) for f in frames]
    rows, start = [], {}
    for i, fr in enumerate(frames):
        g = fr.get('spoof_gt')
        if g is None:
            continue
        g = np.asarray(g, float).reshape(-1)
        cls = 1 if g.size < 8 else int(round(g[7]))
        rid = fr.get('attack_run_id')
        start.setdefault(rid, i)
        v = 10.0 * np.linalg.norm(poses[i][:2, 3] - poses[i - 1][:2, 3]) if i > 0 else np.nan
        iou, lab, sc, n = best_match(preds[i], g[:7])
        t, a = hits({'iou': iou, 'label': lab}, cls)
        r = dict(det=det, segment=seg, frame=i, run_id=str(rid), win_pos=i - start[rid] + 1,
                 range_m=float(np.hypot(g[0], g[1])), n_kept=fr.get('n_spoof_r'), n_cand=fr.get('n_spoof_k'),
                 ego_speed=v, cls=cls, n_pred=n)
        for k, s in enumerate(TT):
            r[f'iou_{s}'] = float(iou[k]); r[f'label_{s}'] = int(lab[k]); r[f'score_{s}'] = float(sc[k])
            r[f'hit_{s}'] = bool(t[k]); r[f'any_{s}'] = bool(a[k])
        rows.append(r)
    lens = {}
    for r in rows:
        lens[r['run_id']] = max(lens.get(r['run_id'], 0), r['win_pos'])
    for r in rows:
        r['win_len'] = lens[r['run_id']]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'frames'))
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = a.runs or sorted(p.parent.parent.name for p in ROOT.glob('*/evaluation/atlas_metrics.json'))
    for run in runs:
        o = out / f'{run}.parquet'
        if o.is_file():
            continue
        m = json.load(open(ROOT / run / 'evaluation' / 'atlas_metrics.json'))
        dets = [k for k, v in m.items() if isinstance(v, dict) and 'per_segment' in v]
        jobs = [(run, d, s) for d in dets for s in sorted(m[d]['per_segment'])]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for rs in ex.map(seg_rows, jobs, chunksize=4) for r in rs]
        df = pd.DataFrame(rows)
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        df.to_parquet(o, index=False)
        chk = {d: round(100 * df[df.det == d]['hit_01'].mean(), 2) for d in dets}
        ref = {d: round(100 * m[d]['asr_target']['0.1'], 2) for d in dets}
        bad = {d: (chk[d], ref[d]) for d in dets if abs(chk[d] - ref[d]) > 0.05}
        print(f'{run}: {len(df)} rows, {len(dets)} dets' + (f'  ASR MISMATCH {bad}' if bad else '  ASR@0.1 matches atlas_metrics'),
              flush=True)


if __name__ == '__main__':
    main()
