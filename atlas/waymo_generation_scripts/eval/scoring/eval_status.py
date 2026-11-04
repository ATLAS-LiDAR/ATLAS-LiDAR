"""Eval coverage of every 50-segment seed-42 dataset (CPU, reads the filesystem only).

    python eval_status.py > <GEN_ROOT>/results/EVAL_STATUS.md

A checkpoint counts as done when its predictions folder holds one file per dataset segment.
Injection studies need the GT-vs-no-GT sweep: CP-SF e9/18/27/30, CP4F e9/18/27/36, PTT e1-6, MSF-4/8 e1-6,
each GT and no-GT. Removal / adverse-baseline studies need the final epoch of the GT models only.
"""
from pathlib import Path

R = Path('/storage/scratch1/9/spanse30/cvpr_datasets')
G = Path('/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/PTT/neurips/generation_runs')
SPLITS = ['global_easy', 'global_medium', 'global_hard', 'rel_fixed_easy', 'rel_fixed_medium', 'rel_fixed_hard']
INJ_EPOCHS = {'CP-SF': ('cp_sf_atlas', [9, 18, 27, 30]), 'CP4F': ('cp4f_atlas', [9, 18, 27, 36]),
              'PTT': ('ptt32_atlas', [2, 4, 6]), 'MSF-4': ('msf4_cvpr', [2, 4, 6]),
              'MSF-8': ('msf8_cvpr', [2, 4, 6])}
FINAL = {'CP-SF': ('cp_sf_atlas', 30), 'CP4F': ('cp4f_atlas', 36), 'MSF-4': ('msf4_cvpr', 6), 'MSF-8': ('msf8_cvpr', 6),
         'PTT': ('ptt32_atlas', 6)}


def find(name):
    for root in (R, G):
        if (root / name).exists():
            return root / name
    return None


def nseg(d):
    p = d / 'dataset'
    return len(list(p.glob('*.pkl'))) if p.is_dir() else 0


def done(d, sub, fam, e, n):
    """predictions complete for (detector folder, family, epoch). MSF headline runs are flat (= GT e6)."""
    ev = d / 'evaluation' / sub
    cands = [ev / f'{fam}_epoch{e}' / 'predictions']
    if sub.startswith('msf') and fam == 'gt' and e == 6:
        cands.append(ev / 'predictions')
    return any(c.is_dir() and len(list(c.glob('*_p.pkl'))) >= n > 0 for c in cands)


def inj_row(label, name, note=''):
    d = find(name)
    if d is None or nseg(d) == 0:
        return f'| {label} | `{name}` | **not generated** | ' + ' | '.join('—' for _ in INJ_EPOCHS) + f' | {note} |', None
    n = nseg(d)
    cells, todo = [], []
    for det, (sub, eps) in INJ_EPOCHS.items():
        g = sum(done(d, sub, 'gt', e, n) for e in eps); ng = sum(done(d, sub, 'nogt', e, n) for e in eps)
        mark = '✅' if g == ng == len(eps) else ('·' if g == ng == 0 else '◐')
        cells.append(f'{mark} {g}/{len(eps)} · {ng}/{len(eps)}')
        todo.append((det, len(eps) - g, len(eps) - ng))
    left = '; '.join(f'{det} ' + ' + '.join(x for x in ((f'GT {g}' if g else ''), (f'no-GT {ng}' if ng else '')) if x)
                     for det, g, ng in todo if g or ng) or 'nothing'
    return f'| {label} | `{name}` | {n} | ' + ' | '.join(cells) + f' | {left} |', todo


def rm_row(label, name, note=''):
    d = find(name)
    if d is None or nseg(d) == 0:
        return f'| {label} | `{name}` | **not generated** | ' + ' | '.join('—' for _ in FINAL) + f' | {note} |', None
    n = nseg(d)
    cells, todo = [], []
    for det, (sub, e) in FINAL.items():
        ok = done(d, sub, 'gt', e, n)
        old = (d / 'evaluation' / {'CP-SF': 'cp_sf', 'MSF-4': 'msf4', 'PTT': 'ptt32'}.get(det, '_') / 'predictions').is_dir()
        cells.append('✅' if ok else ('July only' if old else '·'))
        todo.append((det, 0 if ok else 1))
    left = ', '.join(f'{det} GT' for det, m in todo if m) or 'nothing'
    return f'| {label} | `{name}` | {n} | ' + ' | '.join(cells) + f' | {left} |', todo


def main():
    inj = []
    for s in SPLITS:
        inj.append((f'ATLAS car {s}, adaptive', f'sub50_{s}_seed42', '' if 'medium' not in s else 'view to build'))
    for s in SPLITS:
        inj.append((f'ATLAS car {s}, attenuated', f'sub50_{s}_att_seed42', '' if 'medium' not in s else 'view to build'))
    for m in ('adaptive', 'attenuated'):
        v = '' if m == 'adaptive' else '_att'
        for a in ('global', 'flicker', 'randobj'):
            inj.append((f'ATLAS car hard {a}, {m}', f'abl_car_atlas_hard_{a}{v}_seed42', 'paired car reference' if a == 'global' else ''))
        for a in ('global', 'relative', 'flicker', 'randobj'):
            inj.append((f'ATLAS cyclist {a}, {m}', f'abl_cyc_atlas_{a}{v}_seed42', ''))
    for c in ('car', 'cyc'):
        for a in ('global', 'relative', 'flicker', 'randobj'):
            inj.append((f'paste {c} {a} (real intensity)', f'abl_{c}_paste_{a}_seed42', ''))
    for k, lab in (('jpt', 'per-point jitter only'), ('jfr', 'per-frame jitter only')):
        for s in SPLITS:
            inj.append((f'attenuated + {lab}, {s}', f'{s}_att_{k}_50_seed42', ''))
    for s in ('global_easy', 'global_hard', 'rel_fixed_easy', 'rel_fixed_hard'):
        inj.append((f'trace intensity, {s}', f'sub50_{s}_full_trace_seed42', ''))
    rm = [(f'ATLAS removal az{w}', f'atlas_removal_az{w}_seed42', 'July (rebuttal)' if w in (20, 50) else 'generating now')
          for w in (10, 20, 30, 40, 50, 60)]
    rm += [('baseline: ring/beam dropout 16', 'baseline_ring_16_seed42', ''), ('baseline: ring/beam dropout 32', 'baseline_ring_32_seed42', ''),
           ('baseline: uniform dropout 50%', 'baseline_uniform_50_seed42', ''), ('baseline: uniform dropout 90%', 'baseline_uniform_90_seed42', ''),
           ('baseline: matched sector az20', 'baseline_matched_az20_seed42', ''), ('baseline: matched sector az50', 'baseline_matched_az50_seed42', ''),
           ('control: Gaussian noise σ=0.08', 'gaussian_sigma_008_global_easy_seed42', '')]

    out = ['# Evaluation status: every 50-segment, seed-42 dataset', '',
           'Generated by `cvpr_scripts/eval_status.py` from the files on disk. Injection cells: GT done · no-GT done, out of the '
           'required checkpoints (CP-SF e9/18/27/30, CP4F e9/18/27/36, PTT/MSF e2/4/6). ✅ complete, ◐ partial, · nothing. '
           'MSF no-GT needs the Delta-trained models. Removal cells: final epoch, GT models only; "July only" = a July rebuttal '
           'eval exists but in the old format with CP4F proposals from the PTT fork (to be redone).', '',
           '## Point injection (GT vs no-GT sweep)', '',
           '| Study | Dataset | Segs | CP-SF | CP4F | PTT | MSF-4 | MSF-8 | Left (checkpoints) |', '|---|---|---|---|---|---|---|---|---|']
    tot = {k: [0, 0] for k in INJ_EPOCHS}; gen_missing = []
    for lab, name, note in inj:
        row, todo = inj_row(lab, name, note)
        out.append(row)
        if todo is None:
            gen_missing.append(name); todo = [(k, len(INJ_EPOCHS[k][1]), len(INJ_EPOCHS[k][1])) for k in INJ_EPOCHS]
        for det, g, ng in todo:
            tot[det][0] += g; tot[det][1] += ng
    out += ['', '## Point removal and adverse baselines (final epoch, GT models)', '',
            '| Study | Dataset | Segs | CP-SF e30 | CP4F e36 | MSF-4 e6 | MSF-8 e6 | PTT e6 | Left (final epoch) |', '|---|---|---|---|---|---|---|---|---|']
    rtot = {k: 0 for k in FINAL}
    for lab, name, note in rm:
        row, todo = rm_row(lab, name, note)
        out.append(row)
        if todo is None:
            gen_missing.append(name); todo = [(k, 1) for k in FINAL]
        for det, m in todo:
            rtot[det] += m
    out += ['', '## Remaining work', '',
            f'- Datasets still to generate ({len(gen_missing)}): ' + ', '.join(f'`{x}`' for x in gen_missing), '',
            '| Detector | Injection checkpoint-evals left (GT · no-GT) | Removal evals left |', '|---|---|---|']
    for det in INJ_EPOCHS:
        out.append(f'| {det} | {tot[det][0]} · {tot[det][1]} | {rtot[det]} |')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
