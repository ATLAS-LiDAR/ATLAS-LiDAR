"""Fill the Waymo cells of the main-paper tables from paper_metrics.py outputs. CPU only, no GPU.

    python fill_tables.py --metrics <GEN_ROOT>/results/paper_metrics --tabs <paper>/tabs --out <dir> [--variant fired|argmax|tau_f]

Reads <metrics>/<set>/<det>__<label>.json (+ _instances.pkl), writes <out>/<table>.tex (copies of the paper's tables
with every Waymo cell that has data replaced; cells without data keep their gray placeholder) and <out>/NUMBERS.md
(every number, its CI and where it came from). The paper's own files are never modified.

Conventions (see NUMBERS.md header): variant fired|any|tau_f (the paper's m_tau definition); main-table / factorial / GT-sampling injection =
attenuated intensity (the paper's reference rung); EPR and HR as fractions; mAP = official Waymo L2 mAP (mean of
Vehicle / Pedestrian / Cyclist L2 AP on the full val split, from the training / eval logs) x 100; CIs = 95% paired
bootstrap over the 50 segments (1000 resamples, the same segment draws for every cell); Delta @ matched accuracy =
GT coefficient c of V = a + b * acc + c * GT over all checkpoints of the pair, acc = clean Vehicle AP3D on the
attack frames of the 50 segments (paper_metrics collateral.AP3D_clean; consistent across checkpoints, unlike the
official mAP which exists only for final checkpoints). Pairs with one checkpoint per arm get the final-epoch
difference instead (marked with a dagger).
"""
import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SETTINGS = ['rel_fixed_easy', 'rel_fixed_medium', 'rel_fixed_hard', 'global_easy', 'global_medium', 'global_hard']
ATT = {s: f'sub50_{s}_att_seed42' for s in SETTINGS}
ADA = {s: f'sub50_{s}_seed42' for s in SETTINGS}
TRACE = {'global_easy': 'sub50_global_easy_full_trace_seed42', 'global_hard': 'sub50_global_hard_full_trace_seed42',
         'rel_fixed_easy': 'sub50_rel_fixed_easy_full_trace_seed42', 'rel_fixed_hard': 'sub50_rel_fixed_hard_full_trace_seed42',
         'global_medium': 'global_medium_trace_50_seed42', 'rel_fixed_medium': 'rel_fixed_medium_trace_50_seed42'}
JPT = {s: f'{s}_att_jpt_50_seed42' for s in SETTINGS}
JFR = {s: f'{s}_att_jfr_50_seed42' for s in SETTINGS}
PASTE = {'global_hard': 'abl_car_paste_global_seed42', 'rel_fixed_hard': 'abl_car_paste_relative_seed42'}
AZ = [10, 20, 30, 40, 50, 60]
REM = {a: f'atlas_removal_az{a}_seed42' for a in AZ}

# family -> (det dir, final GT label, final no-GT label)
FAM = {'CP-SF': ('cp_sf_atlas', 'gt_epoch30', 'nogt_epoch30'), 'CP4F': ('cp4f_atlas', 'gt_epoch36', 'nogt_epoch36'),
       'MSF-4f': ('msf4_cvpr', 'gt_epoch6', 'nogt_epoch6'), 'MSF-8f': ('msf8_cvpr', 'gt_epoch6', 'nogt_epoch6'),
       'PTT': ('ptt32_atlas', 'gt_epoch6', 'nogt_epoch6')}
# official full-val Waymo L2 mAP (mean of VEH/PED/CYC L2 AP), x100; source logs in NUMBERS.md
MAP = {('CP-SF', 'gt'): (65.71, 'OpenPCDet centerpoint_singleframe_waymo/default eval/epoch_30/val/cedar_eval/log_eval_20261003-190112.txt'),
       ('CP-SF', 'nogt'): (62.89, 'OpenPCDet centerpoint_singleframe_waymo/no_gt_sampling eval/epoch_30/val/cedar_eval/log_eval_20261003-194453.txt'),
       ('CP4F', 'gt'): (69.06, 'OpenPCDet centerpoint_multiframe_waymo/default eval/epoch_36/val/default/log_eval_20260203-131539.txt'),
       ('CP4F', 'nogt'): (69.86, 'OpenPCDet centerpoint_multiframe_waymo/no_gt_sampling train_20260724-001631.log (EPOCH 36)'),
       ('MSF-4f', 'gt'): (74.74, 'MSF msf_4frames/default log_train_20260220-002511.txt (EPOCH 6)'),
       ('MSF-8f', 'gt'): (74.39, 'MSF msf_8frames/default log_train_20260324-105331.txt (EPOCH 6)'),
       ('MSF-4f', 'nogt'): (74.45, 'msf_nogt/msf_4frames/eval/epoch_6_val/log_metric.txt (Delta)'),
       ('PTT', 'gt'): (74.92, 'PTT output/storage/scratch1/9/spanse30/PTT/tools/cfgs/waymo_models/ptt_32frames/default/train_20260227-142032.log (EPOCH 6; ckpt md5 3d94e5ba = output/cfgs/waymo_models/default/checkpoint_epoch_6.pth used by the evals)'),
       ('PTT', 'nogt'): (74.57, 'PTT ptt_32frames_nogt/no_gt_sampling train_20261010-172729.log (EPOCH 6)')}
PH = re.compile(r'\\ph(?:map|asr|delta|dmap|txt)\b')


class Store:
    def __init__(self, root, variant, reps=1000):
        self.root, self.variant = Path(root), variant
        self.j, self.ps = {}, {}
        for f in sorted(self.root.glob('*/*.json')):
            det, label = f.stem.split('__')
            self.j[(f.parent.name, det, label)] = json.load(open(f))
        segs = set()
        for k in self.j:
            p = self._inst(k)
            if p is not None:
                segs.update(p['seg'].tolist())
        self.segs = np.array(sorted(segs))
        rng = np.random.default_rng(0)
        n = len(self.segs)
        self.W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(reps)]).astype(float)

    def _inst(self, k):
        if k not in self.ps:
            f = self.root / k[0] / f'{k[1]}__{k[2]}_instances.pkl'
            self.ps[k] = pickle.load(open(f, 'rb'))['per_instance'][self.variant] if f.exists() else None
        return self.ps[k]

    def has(self, s, det, label):
        return (s, det, label) in self.j

    def rate(self, s, det, label, tau=None):
        """(point, boot vector) of EPR (injection) / HR (removal) for one set x checkpoint, or None."""
        k = (s, det, label)
        if k not in self.j:
            return None
        var = self.variant if tau is None else self.variant.rsplit('|', 1)[0] + f'|{tau}'
        p = self.ps.get((k, var))
        if p is None:
            f = self.root / s / f'{det}__{label}_instances.pkl'
            p = pickle.load(open(f, 'rb'))['per_instance'][var]
            self.ps[(k, var)] = p
        kind = self.j[k]['kind']
        idx = np.searchsorted(self.segs, p['seg'])
        m, mt = p['m'].astype(float), p['mt'].astype(float)
        if kind == 'injection':
            num, den = np.bincount(idx, mt - m, len(self.segs)), np.bincount(idx, None, len(self.segs)).astype(float)
        else:
            num = np.bincount(idx, m * (1 - mt), len(self.segs)); den = np.bincount(idx, m, len(self.segs))
        point = num.sum() / max(den.sum(), 1)
        boot = (self.W @ num) / np.maximum(self.W @ den, 1)
        return point, boot

    def acc(self, det, label, sets):
        v = [self.j[(s, det, label)]['collateral']['Vehicle']['AP3D_clean'] for s in sets if (s, det, label) in self.j]
        return float(np.mean(v)) if v else None


def mean_over(st, sets, det, label, tau=None):
    r = [st.rate(s, det, label, tau) for s in sets]
    if not r or any(x is None for x in r):
        return None
    return float(np.mean([x[0] for x in r])), np.mean([x[1] for x in r], axis=0)


def ci(b):
    return float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def f2(x):
    return f'{x:.2f}'


def fd(x):
    return f'{x:+.2f}'.replace('+', '$+$').replace('-', '$-$')


def labels_of(st, det, fam, sets):
    """all checkpoints of det/fam ('gt'/'nogt') evaluated on every set in sets."""
    out = []
    for (s, d, l) in st.j:
        if d == det and re.fullmatch(rf'{fam}_epoch\d+', l) and l not in out:
            if all(st.has(x, det, l) for x in sets):
                out.append(l)
    return sorted(out, key=lambda l: int(l.rsplit('epoch', 1)[1]))


def gt_effect(st, fam, sets):
    """Delta @ matched accuracy (regression over checkpoints) or final-epoch difference (dagger)."""
    det, lg, ln = FAM[fam]
    G, N = labels_of(st, det, 'gt', sets), labels_of(st, det, 'nogt', sets)
    if not G or not N:
        return None
    if len(G) + len(N) >= 4 and len(G) >= 2 and len(N) >= 2:
        rows = [(l, 1.0) for l in G] + [(l, 0.0) for l in N]
        acc = np.array([st.acc(det, l, sets) for l, _ in rows])
        V = [mean_over(st, sets, det, l) for l, _ in rows]
        X = np.c_[np.ones(len(rows)), acc, [g for _, g in rows]]
        y = np.array([v[0] for v in V]); Yb = np.stack([v[1] for v in V], 1)        # reps x ckpts
        c = np.linalg.lstsq(X, y, rcond=None)[0][2]
        cb = np.linalg.lstsq(X, Yb.T, rcond=None)[0][2]
        return c, cb, f'regression over {len(G)} GT + {len(N)} no-GT checkpoints', False
    if lg in G and ln in N:
        a, b = mean_over(st, sets, det, lg), mean_over(st, sets, det, ln)
        return a[0] - b[0], a[1] - b[1], 'final-epoch difference', True
    return None


def spearman(a, b):
    def rk(x):
        o = np.argsort(np.argsort(x)); return o.astype(float)
    ra, rb = rk(np.asarray(a)), rk(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])


class Tex:
    def __init__(self, path):
        self.lines = Path(path).read_text().split('\n')
        self.name = Path(path).name

    def fill(self, key, values, occ=0):
        """replace the placeholders of the occ-th line containing key, in order; None keeps the placeholder."""
        hits = [i for i, l in enumerate(self.lines) if key in l]
        if len(hits) <= occ:
            raise KeyError(f'{self.name}: no line #{occ} with {key!r}')
        i = hits[occ]; it = iter(values)

        def rep(m):
            v = next(it, None)
            return m.group(0) if v is None else v
        self.lines[i] = PH.sub(rep, self.lines[i])

    def save(self, out, notes=None):
        """write the table; if notes/<table>.tex exists, insert it right after \\end{table} / \\end{table*}."""
        lines = list(self.lines)
        f = None if notes is None else Path(notes) / self.name
        if f is not None and f.exists():
            k = max(i for i, l in enumerate(lines) if l.strip().startswith('\\end{table'))
            lines[k + 1:k + 1] = ['', f.read_text().rstrip()]
        Path(out).write_text('\n'.join(lines))


def mark_best(cells, lower=True):
    """bold best / underline second; only when every row of the column has a value (a partial column has no 'best')."""
    if any(c is None for c in cells):
        return cells
    vals = [(float(re.sub(r'[^0-9.+-]', '', c.replace('$-$', '-').replace('$+$', '+'))), i) for i, c in enumerate(cells) if c is not None]
    if len(vals) < 2:
        return cells
    vals.sort(reverse=not lower)
    out = list(cells)
    out[vals[0][1]] = r'\textbf{' + out[vals[0][1]] + '}'
    out[vals[1][1]] = r'\underline{' + out[vals[1][1]] + '}'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', required=True, type=Path)
    ap.add_argument('--tabs', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--variant', default='fired|any|tau_f')   # m_tau as defined in sec/5_evaluations.tex (any box of the target class, score >= tau, IoU >= thr)
    ap.add_argument('--notes', type=Path, default=None, help='dir of <table>.tex analysis blocks inserted under each table')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st = Store(a.metrics, a.variant)
    md = [f'# Waymo numbers for the main-paper tables (auto-generated by fill_tables.py)', '',
          f'Variant `{a.variant}`; {len(st.j)} scored (set, detector, checkpoint) files; {len(st.segs)} segments in the bootstrap.',
          'EPR / HR as fractions with 95% segment-bootstrap CIs. Injection = attenuated intensity unless stated. '
          'Empty = no finished eval yet (cell keeps its placeholder).', '']
    nmap = lambda fam, g: f'{MAP[(fam, g)][0]:.1f}' if (fam, g) in MAP else None

    def cell(r):
        return None if r is None else f2(r[0])

    def log(name, r, extra=''):
        if r is None:
            md.append(f'| {name} | — | | {extra} |')
        else:
            lo, hi = ci(r[1]); md.append(f'| {name} | {r[0]:.4f} | [{lo:.4f}, {hi:.4f}] | {extra} |')

    # ---------------- main table
    t = Tex(a.tabs / 'resub_waymo_main.tex')
    md += ['## Table resub_waymo_main (GT-trained detectors, final checkpoints)', '', '| cell | value | 95% CI | note |', '|---|---|---|---|']
    rows = {'CP-SF': r'& CenterPoint~\cite{CenterPoint}   &', 'CP4F': r'& CenterPoint-4f~\cite{CenterPoint} &',
            'MSF-4f': r'& MSF-4f~\cite{He_2023_CVPR}        &', 'MSF-8f': r'& MSF-8f~\cite{He_2023_CVPR}        &',
            'PTT': r'& PTT~\cite{huang2024ptt}           &'}
    cols = {}
    for fam in rows:
        det, lg, _ = FAM[fam]
        v = [nmap(fam, 'gt')]
        for s in SETTINGS:
            r = st.rate(ATT[s], det, lg); v.append(cell(r)); log(f'{fam} EPR {s}', r)
        for az in AZ:
            r = st.rate(REM[az], det, lg); v.append(cell(r)); log(f'{fam} HR az{az}', r)
        cols[fam] = v
    fams = list(rows)
    for j in range(13):
        col = mark_best([cols[f][j] for f in fams], lower=(j > 0))
        for f, c in zip(fams, col):
            cols[f][j] = c
    for fam, key in rows.items():
        t.fill(key, cols[fam])
    t.save(a.out / t.name, a.notes)

    # ---------------- GT sampling table
    t = Tex(a.tabs / 'resub_gt_sampling.tex')
    md += ['', '## Table resub_gt_sampling (Waymo rows)', '', '| cell | value | 95% CI | note |', '|---|---|---|---|']
    keys = {'CP-SF': r'& CenterPoint (CP-SF)~\cite{CenterPoint} &', 'CP4F': r'& CenterPoint-4f~\cite{CenterPoint} &',
            'MSF-4f': r'& MSF-4f~\cite{He_2023_CVPR} &', 'MSF-8f': r'& MSF-8f~\cite{He_2023_CVPR} &', 'PTT': r'& PTT~\cite{huang2024ptt} &'}
    sets_i, sets_r = list(ATT.values()), list(REM.values())
    pooled = defaultdict(list)
    for fam, key in keys.items():
        det, lg, ln = FAM[fam]
        v = [nmap(fam, 'gt'), nmap(fam, 'nogt')]
        for sets, nm in ((sets_i, 'EPR'), (sets_r, 'HR')):
            rg, rn = mean_over(st, sets, det, lg), mean_over(st, sets, det, ln)
            ge = gt_effect(st, fam, sets)
            log(f'{fam} {nm} GT (mean)', rg); log(f'{fam} {nm} no-GT (mean)', rn)
            log(f'{fam} {nm} Delta', None if ge is None else ge[:2], '' if ge is None else ge[2])
            d = None
            if ge is not None:
                lo, hi = ci(ge[1]); dag = r'$^\dagger$' if ge[3] else ''
                d = f'{fd(ge[0])}{dag} {{\\scriptsize[{fd(lo)}, {fd(hi)}]}}'
                pooled[nm].append(ge)
            v += [cell(rg), cell(rn), d]
            if rg is not None: pooled[nm + 'g'].append(rg)
            if rn is not None: pooled[nm + 'n'].append(rn)
        t.fill(key, v)
    t.save(a.out / t.name, a.notes)

    # ---------------- factorial
    t = Tex(a.tabs / 'resub_factorial.tex')
    md += ['', '## Table resub_factorial (Waymo block)', '', '| cell | value | 95% CI | note |', '|---|---|---|---|']
    cells = {}
    for fam in ('CP-SF', 'CP4F'):
        det, lg, ln = FAM[fam]
        for g, lab in (('gt', lg), ('nogt', ln)):
            cells[(fam, g, 'inj')] = mean_over(st, sets_i, det, lab)
            cells[(fam, g, 'gh')] = st.rate(ATT['global_hard'], det, lab)
            cells[(fam, g, 'rem')] = mean_over(st, sets_r, det, lab)
            for m in ('inj', 'gh', 'rem'):
                log(f'{fam} {g} {m}', cells[(fam, g, m)])
    for fam, key in (('CP-SF', '& CenterPoint 1-frame &'), ('CP4F', '& CenterPoint 4-frame &')):
        v = [nmap(fam, 'gt'), nmap(fam, 'nogt')]
        for m in ('inj', 'gh', 'rem'):
            v += [cell(cells[(fam, 'gt', m)]), cell(cells[(fam, 'nogt', m)])]
        t.fill(key, v)

    def diff(x, y):
        return None if x is None or y is None else (x[0] - y[0], x[1] - y[1])

    def dcell(r):
        if r is None:
            return None
        lo, hi = ci(r[1]); return f'{fd(r[0])} {{\\scriptsize[{fd(lo)}, {fd(hi)}]}}'

    def dmap(x, y):
        return None if x is None or y is None else f'{x - y:+.1f}'.replace('+', '$+$').replace('-', '$-$')
    mp = {(f, g): MAP.get((f, g), (None,))[0] for f in ('CP-SF', 'CP4F') for g in ('gt', 'nogt')}
    hist = [dmap(mp[('CP4F', 'gt')], mp[('CP-SF', 'gt')]), dmap(mp[('CP4F', 'nogt')], mp[('CP-SF', 'nogt')])]
    gte = [dmap(mp[('CP-SF', 'gt')], mp[('CP-SF', 'nogt')]), dmap(mp[('CP4F', 'gt')], mp[('CP4F', 'nogt')])]
    inter = [None if None in mp.values() else dmap(mp[('CP4F', 'gt')] - mp[('CP4F', 'nogt')], mp[('CP-SF', 'gt')] - mp[('CP-SF', 'nogt')])]
    for m in ('inj', 'gh', 'rem'):
        h = [diff(cells[('CP4F', g, m)], cells[('CP-SF', g, m)]) for g in ('gt', 'nogt')]
        e = [diff(cells[(f, 'gt', m)], cells[(f, 'nogt', m)]) for f in ('CP-SF', 'CP4F')]
        i = diff(e[1], e[0])
        for n, r in (('history GT', h[0]), ('history noGT', h[1]), ('GT single', e[0]), ('GT multi', e[1]), ('interaction', i)):
            log(f'{m} {n}', r)
        hist += [dcell(h[0]), dcell(h[1])]; gte += [dcell(e[0]), dcell(e[1])]; inter += [dcell(i)]
    t.fill(r'\textit{History effect}', hist); t.fill(r'\textit{GT effect}', gte); t.fill(r'\textit{Interaction}', inter)
    t.save(a.out / t.name, a.notes)

    # ---------------- fidelity ladder (upper block)
    t = Tex(a.tabs / 'resub_fidelity_rank.tex')
    md += ['', '## Table resub_fidelity_rank (upper block)', '',
           'rho = Spearman correlation of detector EPR (mean over the rung\'s settings) with the attenuated reference on the '
           'SAME settings, over every final-checkpoint model (GT and no-GT) evaluated on all of them. '
           'GT effect = mean over pairs with both arms of the final-epoch EPR difference.', '',
           '| rung | rho | 95% CI | GT effect | 95% CI | models / pairs |', '|---|---|---|---|---|---|']
    models = [(f, g) for f in FAM for g in ('gt', 'nogt')]
    rungs = [('Naive paste', PASTE), ('ATLAS, trace intensity', TRACE), ('ATLAS, adaptive intensity', ADA),
             ('ATLAS, attenuated intensity (ref.)', ATT), (r'\quad + per-point jitter', JPT), (r'\quad + per-frame jitter', JFR)]
    for name, rung in rungs:
        sets = list(rung.values()); ref = [ATT[s] for s in rung]
        lab = lambda f, g: FAM[f][1] if g == 'gt' else FAM[f][2]
        ok = [(f, g) for f, g in models if mean_over(st, sets, FAM[f][0], lab(f, g)) is not None
              and mean_over(st, ref, FAM[f][0], lab(f, g)) is not None]
        rho = rcell = rci = None
        if len(ok) >= 4 and 'ref.' not in name:
            R = [mean_over(st, sets, FAM[f][0], lab(f, g)) for f, g in ok]
            F = [mean_over(st, ref, FAM[f][0], lab(f, g)) for f, g in ok]
            rho = spearman([r[0] for r in R], [r[0] for r in F])
            rb = np.array([spearman([r[1][k] for r in R], [r[1][k] for r in F]) for k in range(st.W.shape[0])])
            lo, hi = rci = ci(rb); rcell = f'{rho:.2f} {{\\scriptsize[{lo:.2f}, {hi:.2f}]}}'
        prs = [f for f in FAM if (f, 'gt') in ok and (f, 'nogt') in ok]
        ge = gcell = None
        if prs:
            D = [(mean_over(st, sets, FAM[f][0], FAM[f][1]), mean_over(st, sets, FAM[f][0], FAM[f][2])) for f in prs]
            ge = (float(np.mean([g[0] - n[0] for g, n in D])), np.mean([g[1] - n[1] for g, n in D], axis=0))
            lo, hi = ci(ge[1]); gcell = f'{fd(ge[0])} {{\\scriptsize[{fd(lo)}, {fd(hi)}]}}'
        md.append(f'| {name} | {"" if rho is None else f"{rho:.3f}"} | {"" if rho is None else "[%.3f, %.3f]" % rci} | '
                  f'{"" if ge is None else f"{ge[0]:+.4f}"} | {"" if ge is None else "[%.4f, %.4f]" % ci(ge[1])} | {len(ok)} / {",".join(prs)} |')
        key = name if 'ref.' not in name else 'ATLAS, attenuated intensity (ref.)'
        t.fill(key, [gcell] if 'ref.' in name else [rcell, gcell])
    t.save(a.out / t.name, a.notes)

    # ---------------- mitigation (score threshold, no GT sampling, standard recipe)
    t = Tex(a.tabs / 'resub_mitigation.tex')
    md += ['', '## Table resub_mitigation (rows with data)', '', '| cell | value | 95% CI | note |', '|---|---|---|---|']
    for occ, fam in enumerate(('CP-SF', 'CP4F')):
        det, lg, ln = FAM[fam]
        for key, lab, g, tau in (('& -- & None (standard recipe)', lg, 'gt', None),
                                 (r'& \multirow{4}{*}{Train} & No GT sampling', ln, 'nogt', None),
                                 (r'& \multirow{3}{*}{Test} & Score threshold', lg, 'gt', 0.7)):
            rs = [st.rate(ATT['global_hard'], det, lab, tau), st.rate(ATT['rel_fixed_hard'], det, lab, tau), None,
                  st.rate(REM[20], det, lab, tau), st.rate(REM[50], det, lab, tau)]
            for n, r in zip(('GH', 'RH', 'held-out', 'az20', 'az50'), rs):
                log(f'{fam} {key.split("&")[-1].strip()} {n}', r, '' if tau is None else 'score >= 0.7')
            v = [nmap(fam, g) if tau is None else None] + [cell(r) for r in rs]
            if tau is not None or g == 'nogt':
                v.append('none' if tau is not None else 'retrain')
            t.fill(key, v, occ)
    t.save(a.out / t.name, a.notes)

    md += ['', '## Clean L2 mAP sources', ''] + [f'- {f} {g}: {v:.2f} ({src})' for (f, g), (v, src) in MAP.items()]
    (a.out / 'NUMBERS.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {a.out}: {len(st.j)} metric files used')


if __name__ == '__main__':
    main()
