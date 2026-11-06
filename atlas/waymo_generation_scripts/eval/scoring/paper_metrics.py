"""Paper metrics (sec. Metrics of the resubmission) for ONE dataset x ONE detector checkpoint. CPU only.

    python paper_metrics.py --dataset <GEN_ROOT>/<name> --pred <dataset>/evaluation/<det>/<label>/predictions \
        --clean <clean predictions of the SAME checkpoint> [--kind auto|injection|removal] [--out metrics.json]

--clean: the checkpoint's predictions on UNPERTURBED frames, either an OpenPCDet val result.pkl (list of dicts with
frame_id, e.g. .../eval/epoch_36/val/default/result.pkl) or any pkl holding such a list / {frame_id: dict}
(e.g. the 50-segment clean_recall predictions.pkl). Frames are matched by frame_id.

Per paired instance i (perturbed frame x~_i, same clean frame x_i, target box g_i = phantom box for injection,
tracked victim for removal / baselines / scene-level sets), m(x, g) = 1 if a prediction of the target's class with
score >= tau matches g (argmax-IoU prediction among score >= tau, any class, must have the target class and IoU >=
0.7 vehicle / 0.5 pedestrian, cyclist; the rule of every ATLAS eval). Reported:
  tau_f      calibrated operating point: lowest score threshold whose precision for the target class on the clean
             frames of these segments reaches 0.9 (greedy matching to GT boxes at the class IoU)
  EPR        injection: mean(m~ - m) over attacked frames       (also at fixed tau 0.1 / 0.3 / 0.5 / 0.7)
  HR         removal-type: sum m (1 - m~) / sum m over target-active frames
  raw        raw ASR / raw hiding rate (previous submission)
  track      per attack window (32-frame block): success if matched (injection) / unmatched (removal) in >= 3 of
             any 5 consecutive frames; track-level rate and onset latency (frames from window start)
  collateral C = AP(clean) - AP(perturbed) on the attack frames, Vehicle, own AP3D at IoU 0.7 (not the official
             Waymo L2 mAPH), with the target box excluded from GT and predictions overlapping it ignored
  budget     points added (injection: kept spoof points) or removed per perturbed frame (mean, median)
  CI         95% paired bootstrap over segments (1000 resamples) for EPR, HR
VARIANTS (all computed; keys "<frames>|<rule>|<tau>" in json['variants']):
  frames  fired (attack fired, target present) | all_valid (removal: also target-valid frames where it did not fire)
  rule    argmax (ATLAS eval rule) | any (any target-class box with score >= tau and IoU >= thr, the paper's wording)
  tau     tau_f (all ranges) | tau_f_le50m (calibrated on boxes within 50 m) | 0.1 | 0.3 | 0.5 | 0.7
  track   block_filled|onset_from_block, block_filled|onset_from_first, instances_only
  collateral  per class AP3D, C_vehicle and C_mean3 (mean over the classes present)
Per-instance hits are saved to <out>_instances.pkl for joint bootstraps (paired_diff.py).
Effective robustness needs every detector and is computed from these jsons afterwards (fit V vs clean mAP).
"""
import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_metrics import IOU_THR, iou3d, pred_arrays  # noqa: E402

FIXED = (0.1, 0.3, 0.5, 0.7)
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
PREC = 0.9
K, N = 3, 5


def load_clean(path):
    d = pickle.load(open(path, 'rb'))
    if isinstance(d, dict) and 'det_annos' in d:     # clean_recall/<det>/<label>/predictions.pkl
        d = d['det_annos']
    if isinstance(d, dict) and 'frame_id' not in d:
        return {str(k): v for k, v in d.items()}
    return {str(p['frame_id']): p for p in d}


def target_of(fr):
    g = fr.get('spoof_gt')
    if g is None:
        return None, None
    g = np.asarray(g, np.float64).reshape(-1)
    cls = int(round(g[7])) if g.size == 8 else int(round(g[-1]))
    return g[:7], cls


def cand(pred, gt):
    """(scores, labels, iou with gt) of the predictions that overlap gt at all (others can never match)."""
    boxes, scores, labels = pred_arrays(pred)
    iou = iou3d(boxes, gt)
    k = iou > 0
    return scores[k].astype(np.float32), labels[k].astype(np.int8), iou[k].astype(np.float32)


def match(c, cls, tau, rule):
    """rule 'argmax': the argmax-IoU prediction among score >= tau (any class) must be cls with IoU >= thr
    (the ATLAS eval rule); rule 'any': any prediction of class cls with score >= tau and IoU >= thr."""
    sc, lab, iou = c
    m = sc >= tau
    if not m.any():
        return False
    if rule == 'any':
        return bool(((lab == cls) & (iou >= IOU_THR[cls]) & m).any())
    j = np.flatnonzero(m)[int(np.argmax(iou[m]))]
    return bool(lab[j] == cls and iou[j] >= IOU_THR[cls])


def ap_frames(items, cls=1):
    """AP3D (all-point) over frames: items = [(pred dict, gt boxes (M,7), ignore box or None)]."""
    thr = IOU_THR[cls]
    sc, tp, ngt = [], [], 0
    for pred, gts, ign in items:
        boxes, scores, labels = pred_arrays(pred)
        keep = labels == cls
        boxes, scores = boxes[keep], scores[keep]
        if ign is not None and len(boxes):
            far = iou3d(boxes, ign) < 0.1
            boxes, scores = boxes[far], scores[far]
        ngt += len(gts)
        used = np.zeros(len(gts), bool)
        for j in np.argsort(-scores):
            ok = False
            if len(gts):
                ious = np.array([iou3d(boxes[j:j + 1], g)[0] for g in gts])
                ious[used] = 0
                k = int(np.argmax(ious))
                if ious[k] >= thr:
                    used[k] = True; ok = True
            sc.append(scores[j]); tp.append(ok)
    if ngt == 0:
        return float('nan')
    o = np.argsort(-np.asarray(sc)); tp = np.asarray(tp, float)[o]
    ctp = np.cumsum(tp); rec = ctp / ngt; prec = ctp / np.arange(1, len(tp) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    r = np.r_[0, rec]; p = np.r_[prec[:1] if len(prec) else [0], prec]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def precision_curve(items, cls):
    """precision of class cls at every GRID threshold on clean frames (greedy matching)."""
    tp = np.zeros(len(GRID)); fp = np.zeros(len(GRID))
    for pred, gts in items:
        boxes, scores, labels = pred_arrays(pred)
        keep = labels == cls
        boxes, scores = boxes[keep], scores[keep]
        o = np.argsort(-scores); boxes, scores = boxes[o], scores[o]
        used = np.zeros(len(gts), bool); ok = np.zeros(len(scores), bool)
        for j in range(len(scores)):
            if len(gts):
                ious = np.array([iou3d(boxes[j:j + 1], g)[0] for g in gts]); ious[used] = 0
                k = int(np.argmax(ious))
                if ious[k] >= IOU_THR[cls]:
                    used[k] = True; ok[j] = True
        for t, tau in enumerate(GRID):
            s = scores >= tau
            tp[t] += ok[s].sum(); fp[t] += (~ok[s]).sum()
    return tp / np.maximum(tp + fp, 1)



def segment(args):
    seg_path, pred_dir, clean, kind = args
    frames = pickle.load(open(seg_path, 'rb'))
    preds = pickle.load(open(Path(pred_dir) / f'{seg_path.stem}_p.pkl', 'rb'))
    out = {'inst': [], 'calib': {1: [], 2: [], 3: []}, 'ap': {c: ([], []) for c in (1, 2, 3)}, 'budget': []}
    for i, fr in enumerate(frames):
        fid = str(fr['frame_id']); cp = clean.get(fid)
        gts = np.asarray(fr['gt_boxes'], np.float64)
        attack = (i // 32) % 2 == 1
        if not attack and cp is not None:
            for c in (1, 2, 3):
                out['calib'][c].append((cp, gts[gts[:, -1] == c, :7]))
        g, cls = target_of(fr)
        rm = fr.get('removal_meta') or {}
        active = g is not None and (kind == 'injection' or rm.get('active_this_frame', True))
        if g is None and kind == 'removal' and rm.get('target_box') is not None:   # target valid, attack not fired
            tb = np.asarray(rm['target_box'], np.float64).reshape(-1); g, cls = tb[:7], 1
        if attack and cp is not None:
            for c in (1, 2, 3):
                gc = gts[gts[:, -1] == c, :7]
                if g is not None and kind == 'removal' and c == cls and len(gc):
                    gc = gc[iou3d(gc, g) < 0.5]
                ign = g if (g is not None and c == cls) else None
                out['ap'][c][0].append((cp, gc, ign)); out['ap'][c][1].append((preds[i], gc, ign))
            if kind == 'injection' and fr.get('n_spoof_k') is not None:
                out['budget'].append(int(fr['n_spoof_k']))
            elif rm.get('n_points_before') is not None and rm.get('n_points_after') is not None:
                out['budget'].append(int(rm['n_points_before']) - int(rm['n_points_after']))
        if g is None or cp is None or not attack:
            continue
        out['inst'].append({'seg': seg_path.stem, 'i': i, 'block': i // 32, 'cls': cls, 'active': bool(active),
                            'range': float(np.hypot(g[0], g[1])), 'c_pert': cand(preds[i], g), 'c_clean': cand(cp, g)})
    return out


def track(inst, hit, fill, from_block):
    """per (segment, 32-frame block): success if >= K of any N consecutive frames are hits.
    fill=True: frames of the block without an instance count as non-hits (consecutive in time);
    fill=False: only the instances, in order. Onset latency from the block start or from the first instance."""
    by = {}
    for x, h in zip(inst, hit):
        by.setdefault((x['seg'], x['block']), []).append((x['i'], bool(h)))
    succ, onset = [], []
    for (seg, b), v in by.items():
        v.sort(); idx = np.array([a for a, _ in v]); h = np.array([c for _, c in v], int)
        if fill:
            base = b * 32; seq = np.zeros(32, int); seq[idx - base] = h; off = (idx.min() - base) if not from_block else 0
            seq = seq if from_block else seq[idx.min() - base:]
        else:
            seq = h; off = 0
        win = np.convolve(seq, np.ones(N, int), 'valid') >= K if len(seq) >= N else np.array([seq.sum() >= K])
        succ.append(bool(win.any()))
        if win.any():
            onset.append(int(np.argmax(win)) + K - 1 + (off if not fill else 0))
    return (float(np.mean(succ)) if succ else float('nan'), float(np.median(onset)) if onset else None, len(succ))


def boot(seg_ids, fn, reps=1000, seed=0):
    segs = np.unique(seg_ids); rng = np.random.default_rng(seed)
    by = {s: np.flatnonzero(seg_ids == s) for s in segs}
    est = [fn(np.concatenate([by[s] for s in rng.choice(segs, len(segs))])) for _ in range(reps)]
    return [float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5))]


def calibrate(calib, cls, max_range=None):
    items = calib if max_range is None else [
        (p, g[np.hypot(g[:, 0], g[:, 1]) <= max_range]) for p, g in calib]
    if max_range is not None:   # also drop predictions beyond the range
        items = [({**p, **_keep(p, max_range)}, g) for p, g in items]
    prec = precision_curve(items, cls)
    ok = np.flatnonzero(prec >= PREC)
    return (float(GRID[ok[0]]) if ok.size else float(GRID[-1])), (float(prec[ok[0]]) if ok.size else float(prec[-1])), prec


def _keep(p, r):
    b = np.asarray(p['boxes_lidar']).reshape(len(p['score']), -1) if len(p['score']) else np.zeros((0, 7))
    k = np.hypot(b[:, 0], b[:, 1]) <= r if len(b) else np.zeros(0, bool)
    out = {'boxes_lidar': b[k], 'score': np.asarray(p['score'])[k]}
    if 'pred_labels' in p:
        out['pred_labels'] = np.asarray(p['pred_labels'])[k]
    if 'name' in p:
        out['name'] = np.asarray(p['name'])[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path)
    ap.add_argument('--pred', required=True, type=Path)
    ap.add_argument('--clean', required=True, type=Path)
    ap.add_argument('--kind', default='auto', choices=('auto', 'injection', 'removal'))
    ap.add_argument('--out', type=Path, default=None, help='json path (default <pred>/../paper_metrics.json)')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    segs = sorted((a.dataset / 'dataset').glob('*.pkl'))
    clean = load_clean(a.clean)
    kind = a.kind
    if kind == 'auto':
        f0 = pickle.load(open(segs[0], 'rb'))
        kind = 'removal' if any(f.get('removal_meta') is not None for f in f0) else 'injection'
    by_seg = {}
    for fid, p in clean.items():
        by_seg.setdefault(fid.rsplit('_', 1)[0], {})[fid] = p
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(segment, [(s, a.pred, by_seg.get(s.stem, {}), kind) for s in segs]))
    inst_all = [x for p in parts for x in p['inst']]
    if not inst_all:
        sys.exit('no paired instances (check --clean frame_ids and --pred)')
    cls = int(np.bincount([x['cls'] for x in inst_all if x['active']] or [1]).argmax())
    calib = [it for p in parts for it in p['calib'][cls]]
    taus = {}
    for name, mr in (('tau_f', None), ('tau_f_le50m', 50.0)):
        t, pr, curve = calibrate(calib, cls, mr)
        taus[name] = t
        if mr is None:
            curve_all = curve
    res = {'dataset': a.dataset.name, 'pred': str(a.pred), 'clean': str(a.clean), 'kind': kind, 'target_class': cls,
           'n_segments': len(segs), 'taus': taus, 'precision_target': PREC,
           'precision_curve_allrange': dict(zip(map(str, GRID.tolist()), np.round(curve_all, 4).tolist())),
           'variants': {}}
    frame_sets = {'fired': [x for x in inst_all if x['active']]}
    if kind == 'removal':
        frame_sets['all_valid'] = inst_all
    per_inst = {}
    for fs, inst in frame_sets.items():
        seg_ids = np.array([x['seg'] for x in inst])
        for rule in ('argmax', 'any'):
            for tname, tau in list(taus.items()) + [(str(t), t) for t in FIXED]:
                m = np.array([match(x['c_clean'], x['cls'], tau, rule) for x in inst], bool)
                mt = np.array([match(x['c_pert'], x['cls'], tau, rule) for x in inst], bool)
                key = f'{fs}|{rule}|{tname}'
                per_inst[key] = {'seg': seg_ids, 'i': np.array([x['i'] for x in inst]), 'm': m, 'mt': mt}
                if kind == 'injection':
                    v = float(np.mean(mt.astype(int) - m.astype(int)))
                    rec = {'EPR': v, 'EPR_CI': boot(seg_ids, lambda k: np.mean(mt[k].astype(int) - m[k].astype(int))),
                           'raw_ASR': float(mt.mean()), 'clean_match_rate': float(m.mean())}
                    hit = mt
                else:
                    hid = m & ~mt
                    rec = {'HR': float(hid.sum() / max(m.sum(), 1)),
                           'HR_CI': boot(seg_ids, lambda k: hid[k].sum() / max(m[k].sum(), 1)),
                           'raw_hiding_rate': float((~mt).mean()), 'clean_detected': int(m.sum())}
                    hit = ~mt
                for tv, (fill, fb) in {'block_filled|onset_from_block': (True, True),
                                       'block_filled|onset_from_first': (True, False),
                                       'instances_only': (False, False)}.items():
                    tr = track(inst, hit, fill, fb)
                    rec[f'track[{tv}]'] = {'rate': tr[0], 'onset_median': tr[1], 'n_windows': tr[2]}
                rec['n_instances'] = len(inst)
                res['variants'][key] = rec
    col = {}
    for c in (1, 2, 3):
        cl = [it for p in parts for it in p['ap'][c][0]]; pe = [it for p in parts for it in p['ap'][c][1]]
        col[c] = (ap_frames(cl, c), ap_frames(pe, c))
    nm = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist'}
    res['collateral'] = {nm[c]: {'AP3D_clean': col[c][0], 'AP3D_perturbed': col[c][1], 'C': col[c][0] - col[c][1]}
                         for c in col}
    valid = [c for c in col if not np.isnan(col[c][0])]
    res['collateral']['C_vehicle'] = res['collateral']['Vehicle']['C']
    res['collateral']['C_mean3'] = float(np.mean([res['collateral'][nm[c]]['C'] for c in valid])) if valid else None
    bud = [b for p in parts for b in p['budget']]
    res['budget_points'] = {'mean': float(np.mean(bud)) if bud else None, 'median': float(np.median(bud)) if bud else None}
    res['not_computed'] = ['recovery time (no phantom/target box after a window ends)']
    out = a.out or (a.pred.parent / 'paper_metrics.json')
    json.dump(res, open(out, 'w'), indent=1)
    pickle.dump({'kind': kind, 'per_instance': per_inst}, open(Path(str(out).replace('.json', '_instances.pkl')), 'wb'))
    r = res['variants'][f'fired|argmax|tau_f']
    print(f"{a.dataset.name} [{kind}] tau_f={taus['tau_f']:.2f} " +
          (f"EPR={r['EPR']:.4f} {r['EPR_CI']}" if kind == 'injection' else f"HR={r['HR']:.4f} {r['HR_CI']}") +
          f" C_veh={res['collateral']['C_vehicle']:.4f} C3={res['collateral']['C_mean3']} -> {out}")


if __name__ == '__main__':
    main()
