"""Cross-study comparisons not covered by cvpr_tables.py (CPU; reads atlas_metrics.json / frame records).

    python score_extra.py [--out <GEN_ROOT>/_tables_all]

 X1 intensity_modes.md  adaptive vs attenuated / trace / calibrated / saturated on the same 50 segments,
                        thresholded ASR + paired differences vs adaptive (CP-SF, CP4F, PTT)
 X2 jitter.md           Sato'24 position jitter J0 (attenuated, no jitter) / J1 / J2 / J3 on the same 30
                        segments of global and relative hard; ASR vs the realized box (spoof_gt) and vs the
                        aimed box (spoof_gt_intended), paired differences vs J0
 X3 gt_vs_nogt.md       GT-sampling effect on the 50 GT-vs-no-GT segments of global easy / hard:
                        adaptive vs attenuated vs attenuated + J2 jitter (OLS ASR ~ clean recall + GT over 28
                        checkpoints; final-epoch GT - no-GT paired differences)
"""
import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_metrics import ROOT, THRS, best_match, hits, paired_bootstrap  # noqa: E402

OLDG = Path('/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/PTT/neurips/generation_runs')
DETS = [('CP-SF', 'cp_sf_atlas/gt_epoch30'), ('CP4F', 'cp4f_atlas/gt_epoch36'), ('PTT', 'ptt32_atlas/gt_epoch6')]
EPOCHS = {'CP-SF': ('cp_sf_atlas', 'cp_sf', [9, 18, 27, 30]), 'CP4F': ('cp4f_atlas', 'cp4f', [9, 18, 27, 36]),
          'PTT-32': ('ptt32_atlas', 'ptt32', [1, 2, 3, 4, 5, 6])}
DASH = '—'


def am(run):
    p = ROOT / run / 'evaluation' / 'atlas_metrics.json'
    return json.load(open(p)) if p.is_file() else {}


def restrict(m, segs):
    """ASR (target) at each threshold from per-segment counts restricted to segs."""
    rows = [m['per_segment'][s] for s in segs if s in m['per_segment']]
    A = np.array(rows, float)
    T = (A.shape[1] - 1) // 2
    return {'n': int(A[:, 0].sum()), 'asr': A[:, 1:1 + T].sum(0) / A[:, 0].sum(),
            'per_segment': {s: m['per_segment'][s] for s in segs if s in m['per_segment']}}


def thr(v):
    return '/'.join(f'{100 * x:.1f}' for x in v)


def dcell(a_seg, b_seg):
    d, lo, hi = paired_bootstrap(a_seg, b_seg)
    return f'{100 * d[0]:+.1f} [{100 * lo[0]:+.1f},{100 * hi[0]:+.1f}] · {100 * d[3]:+.1f} [{100 * lo[3]:+.1f},{100 * hi[3]:+.1f}]'


# ------------------------------------------------------------ X1 intensity modes
def x1():
    md = ['# X1 Intensity modes on the same 50 segments (ASR % at 0.1/0.3/0.5/0.7; Δ vs adaptive, paired, at 0.1 · 0.7)', '']
    modes = lambda s: [('adaptive', f'sub50_{s}_seed42'), ('attenuated', f'sub50_{s}_att_seed42'),  # noqa: E731
                       ('trace', f'sub50_{s}_full_trace_seed42'), ('calibrated', f'sub50_{s}_full_calib_seed42'),
                       ('saturated', f'{s}_sat_seed42')]
    for s in ('global_easy', 'global_hard', 'rel_fixed_easy', 'rel_fixed_hard'):
        md += [f'## {s}', '', '| Mode | ' + ' | '.join(d for d, _ in DETS) + ' | Δ vs adaptive: ' + ' | '.join(d for d, _ in DETS) + ' |',
               '|---|' + '---|' * (2 * len(DETS))]
        base = am(f'sub50_{s}_seed42')
        for mode, run in modes(s):
            M = am(run)
            if not M:
                continue
            segs = sorted(M.get(DETS[0][1], {}).get('per_segment', {}))
            cells, dels = [], []
            for _, k in DETS:
                if k in M and k in base:
                    cells.append(thr(restrict(M[k], segs)['asr']))
                    dels.append(DASH if mode == 'adaptive' else dcell(restrict(M[k], segs)['per_segment'], restrict(base[k], segs)['per_segment']))
                else:
                    cells.append(DASH); dels.append(DASH)
            md.append(f'| {mode} | ' + ' | '.join(cells) + ' | ' + ' | '.join(dels) + ' |')
        md.append('')
    return md


# ------------------------------------------------------------ X2 jitter
def _intended_hits(args):
    ds_file, pred_file = args
    frames = pickle.load(open(ds_file, 'rb'))
    preds = pickle.load(open(pred_file, 'rb'))
    rows = []
    for i, fr in enumerate(frames):
        g = fr.get('spoof_gt_intended')
        if fr.get('spoof_gt') is None or g is None:
            continue
        iou, lab, _, _ = best_match(preds[i], np.asarray(g, float)[:7])
        t, _ = hits({'iou': iou, 'label': lab}, 1)
        rows.append(t.astype(int))
    return np.array(rows).sum(0) if rows else np.zeros(len(THRS)), len(rows)


def x2(workers):
    md = ['# X2 Spoof position jitter (attenuated; Sato NDSS\'24 along-ray error), 30 segments', '',
          'J0 = no jitter (same 30 segments of <split>_full_att_seed42). J1: σ_inner 0.10 m; J2 (Sato): 0.10 + σ_inter 0.35 m; '
          'J3: 0.20 + 0.70 m. ASR % at 0.1/0.3/0.5/0.7 vs the realized box (main) and vs the aimed box; Δ vs J0 paired at 0.1 · 0.7.', '']
    for s in ('global_hard', 'rel_fixed_hard'):
        segs = sorted(json.load(open(ROOT / f'{s}_att_j1_seed42' / 'evaluation' / 'atlas_metrics.json'))[DETS[0][1]]['per_segment'])
        J0 = am(f'{s}_full_att_seed42')
        md += [f'## {s}', '', '| Detector | Level | ASR (realized box) | ASR (aimed box) | Δ vs J0 (realized) |', '|---|---|---|---|---|']
        for name, k in DETS:
            if k not in J0:
                continue
            r0 = restrict(J0[k], segs)
            md.append(f'| {name} | J0 | {thr(r0["asr"])} | = | — |')
            for j in (1, 2, 3):
                run = f'{s}_att_j{j}_seed42'
                M = am(run)
                if k not in M:
                    continue
                rj = restrict(M[k], segs)
                jobs = [(ROOT / run / 'dataset' / f'{g}.pkl', ROOT / run / 'evaluation' / k / 'predictions' / f'{g}_p.pkl') for g in segs]
                with ProcessPoolExecutor(workers) as ex:
                    res = list(ex.map(_intended_hits, jobs))
                hit = sum(r[0] for r in res); n = sum(r[1] for r in res)
                md.append(f'| {name} | J{j} | {thr(rj["asr"])} | {thr(hit / max(n, 1))} | {dcell(rj["per_segment"], r0["per_segment"])} |')
        md.append('')
    return md


# ------------------------------------------------------------ X3 GT vs no-GT three ways
def clean_recall(rsub, label):
    p = OLDG / 'clean_recall' / rsub / label / 'recall.json'
    return json.loads(p.read_text())['recall_vehicle_iou0.7'] if p.is_file() else None


def x3():
    from scipy import stats
    md = ['# X3 GT sampling vs no GT sampling: adaptive vs attenuated vs attenuated + J2 jitter (50 segments)', '',
          'OLS over 28 checkpoints: ASR(0.1) ~ clean vehicle recall + GT; GT effect = ASR points at equal recall (95% CI, optimistic: '
          'epochs of one run are not independent). Final-epoch Δ = GT − no-GT, paired, at 0.1 · 0.7.', '',
          '| Split | Version | n | R² GT | R² no-GT | GT effect [95% CI] | Δ final CP-SF | Δ final CP4F | Δ final PTT |',
          '|---|---|---|---|---|---|---|---|---|']
    for s in ('global_easy', 'global_hard', 'rel_fixed_easy', 'rel_fixed_hard'):
        for ver, run in (('adaptive', f'sub50_{s}_seed42'), ('attenuated', f'sub50_{s}_att_seed42'),
                         ('attenuated + J2', f'{s}_att_j2_50_seed42')):
            M = am(run)
            if not M:
                continue
            rows, fin = [], []
            for det, (sub, rsub, eps) in EPOCHS.items():
                for fam in ('gt', 'nogt'):
                    for e in eps:
                        m = M.get(f'{sub}/{fam}_epoch{e}')
                        r = clean_recall(rsub, f'{fam}_epoch{e}')
                        if m and r is not None:
                            rows.append((100 * r, 100 * m['asr_target']['0.1'], fam == 'gt'))
                g, n = M.get(f'{sub}/gt_epoch{eps[-1]}'), M.get(f'{sub}/nogt_epoch{eps[-1]}')
                fin.append(dcell(g['per_segment'], n['per_segment']) if g and n else DASH)
            if len(rows) < 6:
                md.append(f'| {s} | {ver} | {len(rows)} | — | — | — | ' + ' | '.join(fin) + ' |')
                continue
            R = np.array(rows, float)
            r2 = []
            for gflag in (1, 0):
                sel = R[R[:, 2] == gflag]
                r2.append(stats.linregress(sel[:, 0], sel[:, 1]).rvalue ** 2 if len(sel) >= 3 else np.nan)
            X = np.c_[np.ones(len(R)), R[:, 0], R[:, 2]]
            beta, *_ = np.linalg.lstsq(X, R[:, 1], rcond=None)
            res = R[:, 1] - X @ beta
            dof = len(R) - 3
            se = np.sqrt((res @ res / dof) * np.linalg.inv(X.T @ X)[2, 2])
            tq = stats.t.ppf(0.975, dof)
            md.append(f'| {s} | {ver} | {len(R)} | {r2[0]:.2f} | {r2[1]:.2f} | {beta[2]:+.1f} [{beta[2] - tq * se:+.1f},{beta[2] + tq * se:+.1f}] | '
                      + ' | '.join(fin) + ' |')
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / '_tables_all'))
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (('intensity_modes.md', x1), ('jitter.md', lambda: x2(a.workers)), ('gt_vs_nogt.md', x3)):
        try:
            md = fn()
        except Exception as e:  # keep going; report which table failed
            md = [f'# {name}: FAILED: {type(e).__name__}: {e}']
        (out / name).write_text('\n'.join(md) + '\n')
        print('wrote', out / name)


if __name__ == '__main__':
    main()
