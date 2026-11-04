"""Score-thresholded ASR (and per-frame match records) for ANY ATLAS run and eval label. CPU only.

    python atlas_metrics.py <run> [<run> ...] [--workers 16] [--only cp4f_atlas/gt_epoch36 ...]

<run> is a folder under $GEN_ROOT (default scratch cvpr_datasets), e.g. global_hard_full_seed42,
sub50_global_easy_seed42, abl_cyc_atlas_flicker_seed42. Every finished eval in <run>/evaluation
is scored (cp_sf_atlas/<label>, cp4f_atlas/<label>, ptt32_atlas/<label>, msf4_cvpr, msf8_cvpr;
finished = summary.pkl or asr_only.json and one prediction file per segment).

Rule (identical to the eval scripts' spoof_false_positive): per attacked frame (spoof_gt present)
keep the predictions with score >= t, take the argmax-3D-IoU prediction of ANY class;
  target-class hit: its label is the spoof's class (spoof_gt[7]; Vehicle unless a cyclist set)
                    and IoU >= 0.7 (Vehicle) / 0.5 (Pedestrian, Cyclist)
  any-class hit   : its label is any class, with that class's IoU threshold
t in 0.1 / 0.3 / 0.5 / 0.7; t = 0.1 reproduces the eval's own ASR (checked and recorded).
Frames with no prediction count as misses (denominator = all attacked frames, as in the evals).

Outputs per eval label <run>/evaluation/<det>/<label>/frame_records.pkl
  {segment: {local_idx: {'iou','label','score': arrays over thresholds, 'n_pred'}}}
and per run <run>/evaluation/atlas_metrics.json with, per label: target / any-class / per-class ASR
at each t, segment-mean ASR, per-segment counts (for paired bootstraps), phantom scores and
best IoUs at t = 0.1, the eval ASR check and L2 mAP/mAPH. Stage 0 (once per run) reads only
spoof_gt / pose / attack ids from dataset/*.pkl into evaluation/spoof_gt_cache.pkl (the k-history
analysis uses it too); sub50_/sub30_ views take theirs from the parent split.
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

ROOT = Path(os.environ.get('GEN_ROOT', '/storage/scratch1/9/spanse30/cvpr_datasets'))
THRS = (0.1, 0.3, 0.5, 0.7)
IOU_THR = {1: 0.7, 2: 0.5, 3: 0.5}              # Vehicle, Pedestrian, Cyclist
CLASS_NAME = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist'}
NAME_ID = {v: k for k, v in CLASS_NAME.items()}
DET_SUBS = ('cp_sf_atlas', 'cp4f_atlas', 'ptt32_atlas')
SINGLE_LABEL_DETS = ('msf4_cvpr', 'msf8_cvpr')
ALLOW_NO_META = False


# ---------------------------------------------------------------- geometry
def _poly(b):
    x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
    c, s = np.cos(yaw), np.sin(yaw)
    return Polygon([(x + c * dx - s * dy, y + s * dx + c * dy)
                    for dx, dy in ((l / 2, w / 2), (-l / 2, w / 2), (-l / 2, -w / 2), (l / 2, -w / 2))])


def iou3d(preds, gt):
    """preds (N, >=7), gt (>=7,) -> (N,) rotated 3D IoU (BEV polygon intersection x height overlap)."""
    out = np.zeros(len(preds))
    if len(preds) == 0:
        return out
    g = _poly(gt)
    gz0, gz1, gv = gt[2] - gt[5] / 2, gt[2] + gt[5] / 2, gt[3] * gt[4] * gt[5]
    reach = (np.hypot(preds[:, 3], preds[:, 4]) + np.hypot(gt[3], gt[4])) / 2
    near = np.hypot(preds[:, 0] - gt[0], preds[:, 1] - gt[1]) <= reach
    for i in np.flatnonzero(near):
        b = preds[i]
        h = max(0.0, min(b[2] + b[5] / 2, gz1) - max(b[2] - b[5] / 2, gz0))
        if h <= 0:
            continue
        iv = _poly(b).intersection(g).area * h
        out[i] = iv / (b[3] * b[4] * b[5] + gv - iv + 1e-9)
    return out


def pred_arrays(p):
    boxes = np.asarray(p.get('boxes_lidar', np.zeros((0, 7))), np.float64)
    boxes = boxes.reshape(len(boxes), -1) if boxes.size else np.zeros((0, 7))
    scores = np.asarray(p.get('score', np.zeros(0)), np.float64).reshape(-1)
    if 'pred_labels' in p and len(np.asarray(p['pred_labels']).reshape(-1)) == len(scores):
        labels = np.asarray(p['pred_labels'], np.int64).reshape(-1)
    else:
        labels = np.asarray([NAME_ID.get(str(n), 0) for n in p.get('name', [])], np.int64)
    return boxes, scores, labels


def best_match(pred, gt):
    """-> (iou, label, score) arrays over THRS for the argmax-IoU prediction among score >= t."""
    boxes, scores, labels = pred_arrays(pred)
    iou = iou3d(boxes, gt)
    out = np.zeros((3, len(THRS)))
    for k, t in enumerate(THRS):
        m = np.flatnonzero(scores >= t)
        if m.size:
            j = m[int(np.argmax(iou[m]))]           # first max, as torch.argmax in the evals
            out[:, k] = (iou[j], labels[j], scores[j])
    return out[0], out[1].astype(np.int64), out[2], len(scores)


def hits(rec, cls):
    """target-class and any-class hit arrays (over THRS) for one frame record."""
    thr_lab = np.array([IOU_THR.get(int(l), np.inf) for l in rec['label']])
    anyc = (rec['label'] > 0) & (rec['iou'] >= thr_lab)
    return anyc & (rec['label'] == cls), anyc


# ---------------------------------------------------------------- spoof cache
def _read_spoof(path):
    with open(path, 'rb') as f:
        frames = pickle.load(f)
    out = {}
    for i, fr in enumerate(frames):
        g = fr.get('spoof_gt')
        g = None if g is None else np.asarray(g, np.float64).reshape(-1)
        out[i] = {'gt': None if g is None else g[:7], 'cls': 1 if g is None or g.size < 8 else int(round(g[7])),
                  'n_k': fr.get('n_spoof_k'), 'n_r': fr.get('n_spoof_r'),
                  'pose': np.asarray(fr['pose'], np.float64), 'run': fr.get('attack_run_id')}
    return path.stem, out


def parent_of(run):
    for pre, n in (('sub50_', 50), ('sub30_', 30)):
        if run.startswith(pre):
            return run[len(pre):].replace('_seed42', '_full_seed42')
    return None


def spoof_cache(run, workers):
    R = ROOT / run
    p = R / 'evaluation' / 'spoof_gt_cache.pkl'
    segs = sorted(x.name[:-6] for x in (R / 'annos').glob('*_a.pkl'))
    if p.is_file():
        cache = pickle.load(open(p, 'rb'))
        if set(segs) <= set(cache):
            return {s: cache[s] for s in segs}
    par = parent_of(run)
    if par and (ROOT / par / 'evaluation' / 'spoof_gt_cache.pkl').is_file() and \
            set(segs) <= set(pickle.load(open(ROOT / par / 'evaluation' / 'spoof_gt_cache.pkl', 'rb'))):
        cache = spoof_cache(par, workers)
        sub = {s: cache[s] for s in segs}
    else:                                            # no (complete) parent cache: read the view's own files
        files = [R / 'dataset' / f'{s}.pkl' for s in segs]
        with ProcessPoolExecutor(workers) as ex:
            sub = dict(ex.map(_read_spoof, files))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    pickle.dump(sub, open(tmp, 'wb'))
    os.replace(tmp, p)
    return sub


# ---------------------------------------------------------------- evals
def discover(run, segs):
    E = ROOT / run / 'evaluation'
    found = {}
    cands = [(f'{d}/{l.name}', l) for d in DET_SUBS if (E / d).is_dir() for l in sorted((E / d).iterdir())
             if l.is_dir() and l.name != 'cp4f_proposals'] + [(d, E / d) for d in SINGLE_LABEL_DETS]
    for key, d in cands:
        meta = d / 'metrics.json' if (d / 'summary.pkl').is_file() else d / 'asr_only.json'
        if not meta.is_file() and not ALLOW_NO_META:
            continue
        n = sum((d / 'predictions' / f'{s}_p.pkl').is_file() for s in segs)
        if n == len(segs):
            found[key] = d
    return found


def _score_segment(args):
    pred_file, spoof = args
    preds = pickle.load(open(pred_file, 'rb'))
    recs = {}
    for i, r in spoof.items():
        if r['gt'] is None:
            continue
        iou, lab, sc, n = best_match(preds[i], r['gt'])
        recs[i] = {'iou': iou, 'label': lab, 'score': sc, 'n_pred': n}
    return recs


def frame_records(d, spoof, workers, force=False):
    out = d / 'frame_records.pkl'
    segs = sorted(spoof)
    newest = max((d / 'predictions' / f'{s}_p.pkl').stat().st_mtime for s in segs)
    if out.is_file() and not force and out.stat().st_mtime > newest:
        recs = pickle.load(open(out, 'rb'))
        if set(segs) <= set(recs):
            return {s: recs[s] for s in segs}
    jobs = [(d / 'predictions' / f'{s}_p.pkl', spoof[s]) for s in segs]
    with ProcessPoolExecutor(workers) as ex:
        recs = dict(zip(segs, ex.map(_score_segment, jobs)))
    tmp = out.with_suffix('.tmp')
    pickle.dump(recs, open(tmp, 'wb'))
    os.replace(tmp, out)
    return recs


def summarize(recs, spoof, d):
    T = len(THRS)
    per_seg, fp_scores, best_iou, cls_seen = {}, [], [], set()
    tot = np.zeros(T)
    totc = {c: np.zeros(T) for c in CLASS_NAME}
    n_all = 0
    for s, fr in recs.items():
        ht, ha = np.zeros(T, int), np.zeros(T, int)
        for i, r in fr.items():
            cls = spoof[s][i].get('cls', 1)     # caches built before 2026-10-29 hold vehicle spoofs only
            cls_seen.add(cls)
            t_hit, a_hit = hits(r, cls)
            ht += t_hit
            ha += a_hit
            for c in CLASS_NAME:
                totc[c] += a_hit & (r['label'] == c)
            best_iou.append(float(r['iou'][0]))
            if t_hit[0]:
                fp_scores.append(float(r['score'][0]))
        per_seg[s] = [len(fr)] + ht.tolist() + ha.tolist()
        n_all += len(fr)
    A = np.array(list(per_seg.values()), float)
    n = A[:, 0]
    tgt, anyc = A[:, 1:1 + T].sum(0) / max(n.sum(), 1), A[:, 1 + T:].sum(0) / max(n.sum(), 1)
    seg_mean = (A[n > 0, 1:1 + T] / n[n > 0, None]).mean(0) if (n > 0).any() else np.zeros(T)
    cls = sorted(cls_seen)[0] if len(cls_seen) == 1 else None
    mp = d / 'metrics.json' if (d / 'metrics.json').is_file() else d / 'asr_only.json'
    meta = json.load(open(mp)) if mp.is_file() else {}   # --allow_no_meta: predictions only, no eval ASR check / L2
    tm = meta.get('target_metrics', {})
    ref = tm.get('asr') if cls in (None, 1) else tm.get({2: 'asr_pedestrian', 3: 'asr_cyclist'}[cls])
    l2 = meta.get('l2_summary') or {}
    out = {'n_target_frames': int(n_all), 'spoof_class': CLASS_NAME.get(cls, 'mixed'),
           'asr_target': dict(zip(map(str, THRS), map(float, tgt))),
           'asr_any': dict(zip(map(str, THRS), map(float, anyc))),
           'asr_by_class': {CLASS_NAME[c]: dict(zip(map(str, THRS), map(float, v / max(n_all, 1)))) for c, v in totc.items()},
           'asr_target_segment_mean': dict(zip(map(str, THRS), map(float, seg_mean))),
           'eval_asr_ref': ref, 'match_eval_at_0.1': None if ref is None else bool(abs(tgt[0] - ref) < 2e-3),
           'l2_map': l2.get('l2_map'), 'l2_maph': l2.get('l2_maph'),
           'recall': meta.get('recall'),
           'per_segment': per_seg, 'phantom_scores_at_0.1': fp_scores, 'best_iou_at_0.1': best_iou}
    return out


def run_metrics(run, workers, only=None, force=False):
    spoof = spoof_cache(run, workers)
    segs = sorted(spoof)
    evals = discover(run, segs)
    if only:
        evals = {k: v for k, v in evals.items() if k in only}
    jpath = ROOT / run / 'evaluation' / 'atlas_metrics.json'
    table = json.load(open(jpath)) if jpath.is_file() else {}
    for key, d in evals.items():
        recs = frame_records(d, spoof, workers, force)
        table[key] = summarize(recs, spoof, d)
        m = table[key]
        print(f'{run:34s} {key:34s} target ' + ' '.join(f'{100 * m["asr_target"][str(t)]:5.1f}' for t in THRS)
              + f' | any@0.1 {100 * m["asr_any"]["0.1"]:5.1f} | eval {m["eval_asr_ref"] and 100 * m["eval_asr_ref"]:} '
              + ('OK' if m['match_eval_at_0.1'] else 'CHECK') + f' | L2 {m["l2_map"]}', flush=True)
    tmp = jpath.with_suffix('.tmp')
    json.dump(table, open(tmp, 'w'))
    os.replace(tmp, jpath)
    return table


def paired_bootstrap(a_seg, b_seg, reps=5000, seed=0, col=1):
    """ASR(a) - ASR(b) and its 95% CI from per-segment counts ([n, hits...]); same frames in a, b."""
    segs = sorted(set(a_seg) & set(b_seg))
    A = np.array([a_seg[s] for s in segs], float)
    B = np.array([b_seg[s] for s in segs], float)
    n = A[:, 0]
    T = (A.shape[1] - 1) // 2
    sl = slice(col, col + T)
    d = (A[:, sl] - B[:, sl]).sum(0) / n.sum()
    idx = np.random.default_rng(seed).integers(0, len(segs), (reps, len(segs)))
    bs = (A[idx][:, :, sl] - B[idx][:, :, sl]).sum(1) / n[idx].sum(1, keepdims=True)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return d, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--only', nargs='*', default=None, help='eval keys, e.g. cp4f_atlas/gt_epoch36 msf4_cvpr')
    ap.add_argument('--force', action='store_true', help='recompute frame_records.pkl')
    ap.add_argument('--allow_no_meta', action='store_true',
                    help='also score evals with complete predictions but no metrics file (e.g. linked headline preds on a view)')
    a = ap.parse_args()
    global ALLOW_NO_META
    ALLOW_NO_META = a.allow_no_meta
    bad = 0
    for run in a.runs:
        t = run_metrics(run, a.workers, a.only, a.force)
        bad += sum(1 for v in t.values() if v.get('match_eval_at_0.1') is False)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
