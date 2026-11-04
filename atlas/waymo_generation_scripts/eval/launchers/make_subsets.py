"""Build the rebuttal-size views of the regenerated 195-segment splits (symlinks only, seconds).

    python make_subsets.py            # build / refresh all subsets (idempotent)
    python make_subsets.py --check    # verify only

Why: every 50-segment rebuttal set (global/rel_fixed easy/hard and all abl_* sets) uses the same 50
segments, the seed-42 sample np.random.default_rng(42).choice(195, 50) that the rebuttal's
eval-only metrics also used; the 30-segment schedule sets (global_medium / seq_ / intermittent06_)
share their own 30. Those segments of the 195-segment splits hold the SAME frames (point sets and
spoof_gt identical, checked 2026-10-29), so a subset view of the 195-segment split stands in for
the separate 50-/30-segment datasets and the headline predictions of the main wave are reused.

    sub50_<split>_seed42   split in global_easy, global_hard, rel_fixed_easy, rel_fixed_hard
    sub30_global_medium_seed42

Each view: dataset/<seg>.pkl, annos/<seg>_a.pkl, metrics/frames/<seg>_frames.csv (+ manifest)
symlinked from <split>_full_seed42; evaluation/ is a REAL directory in which only finished
headline evals of the parent are linked per segment (predictions/<seg>_p.pkl), plus the
parent's CP4F proposal caches. Evals run on the view write their own summary/metrics (subset L2)
into the view; a symlinked prediction that an eval rewrites is replaced, never the parent file.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(os.environ.get('GEN_ROOT', '/storage/scratch1/9/spanse30/cvpr_datasets'))
SUBSETS = {f'sub50_{s}_seed42': (f'{s}_full_seed42', f'{s}_seed42')
           for s in ('global_easy', 'global_hard', 'rel_fixed_easy', 'rel_fixed_hard')}
SUBSETS['sub30_global_medium_seed42'] = ('global_medium_full_seed42', 'global_medium_seed42')
# VARIANT=_att (env): ONLY the views of the variant's 195-segment splits, named like the adaptive ones
# (sub50_<split>_att_seed42 <- <split>_full_att_seed42, sub30_global_medium_att_seed42)
_V = os.environ.get('VARIANT', '')
if _V:
    SUBSETS = {f'sub50_{s}{_V}_seed42': (f'{s}_full{_V}_seed42', f'{s}_seed42')
               for s in ('global_easy', 'global_hard', 'rel_fixed_easy', 'rel_fixed_hard')}
    SUBSETS[f'sub30_global_medium{_V}_seed42'] = (f'global_medium_full{_V}_seed42', 'global_medium_seed42')
# medium splits on the same 50 segments (the 50-seg reference is any 50-seg rebuttal set: global_easy_seed42;
# global_medium_seed42 is the 30-segment schedule set and must not be used as the reference)
for s in ('global_medium', 'rel_fixed_medium'):
    SUBSETS[f'sub50_{s}{_V}_seed42'] = (f'{s}_full{_V}_seed42', 'global_easy_seed42')
# intensity-assumption ablations (trace replay / calibrated once per attack), same 50 segments
for s in (() if _V else ('global_easy', 'global_hard', 'rel_fixed_easy', 'rel_fixed_hard')):
    for m in ('trace', 'calib'):
        SUBSETS[f'sub50_{s}_full_{m}_seed42'] = (f'{s}_full_{m}_seed42', f'{s}_seed42')
# headline evals of the main wave whose per-segment predictions are reused (det_subdir/label)
HEADLINE = ('cp_sf_atlas/gt_epoch30', 'cp4f_atlas/gt_epoch36', 'ptt32_atlas/gt_epoch6',
            'msf4_cvpr', 'msf8_cvpr')
CACHES = ('ptt32_atlas/cp4f_proposals/gt_epoch36', 'msf_cp4f_proposals/gt_epoch36')


def link(src, dst):
    """Symlink dst -> src; a correct link is kept, a real file is never touched."""
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.is_symlink():
        if os.readlink(dst) == str(src):
            return 0
        dst.unlink()
    elif dst.exists():
        return 0                                     # real file written by an eval on the view
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)
    return 1


def segments_of(run):
    return sorted(p.name[:-6] for p in (ROOT / run / 'annos').glob('*_a.pkl'))


def build(name, parent, ref, check_only):
    P, V = ROOT / parent, ROOT / name
    segs = segments_of(ref)
    full = set(segments_of(parent))
    missing = [s for s in segs if s not in full]
    if not segs or missing:
        sys.exit(f'{name}: {len(segs)} segments in {ref}, {len(missing)} not in {parent}')
    new = 0
    if not check_only:
        for s in segs:
            new += link(P / 'dataset' / f'{s}.pkl', V / 'dataset' / f'{s}.pkl')
            new += link(P / 'annos' / f'{s}_a.pkl', V / 'annos' / f'{s}_a.pkl')
            new += link(P / 'metrics' / 'frames' / f'{s}_frames.csv', V / 'metrics' / 'frames' / f'{s}_frames.csv')
        new += link(P / 'metrics' / 'run_manifest.json', V / 'metrics' / 'run_manifest.json')
        for h in HEADLINE:
            if not (P / 'evaluation' / h / 'summary.pkl').is_file():
                continue                             # parent eval not finished: link on a later run
            for s in segs:
                new += link(P / 'evaluation' / h / 'predictions' / f'{s}_p.pkl',
                            V / 'evaluation' / h / 'predictions' / f'{s}_p.pkl')
        for c in CACHES:
            if (P / 'evaluation' / c).is_dir():
                for s in segs:
                    f = P / 'evaluation' / c / f'{s}_cp4f.pkl'
                    if f.is_file():
                        new += link(f, V / 'evaluation' / c / f'{s}_cp4f.pkl')
        pc = P / 'evaluation' / 'spoof_gt_cache.pkl'
        vc = V / 'evaluation' / 'spoof_gt_cache.pkl'
        if pc.is_file() and not vc.is_file():
            cache = pickle.load(open(pc, 'rb'))
            pickle.dump({s: cache[s] for s in segs}, open(vc, 'wb'))
    nd = len(list((V / 'dataset').glob('*.pkl')))
    na = len(list((V / 'annos').glob('*_a.pkl')))
    links = {h: len(list((V / 'evaluation' / h / 'predictions').glob('*_p.pkl'))) for h in HEADLINE}
    ok = nd == na == len(segs) and set(segments_of(name)) == set(segs)
    print(f'{name:30s} <- {parent:28s} {len(segs)} segs | dataset {nd} annos {na} | new links {new} | '
          'headline preds ' + ' '.join(f'{h.split("/")[0]}:{n}' for h, n in links.items()) + (' OK' if ok else ' MISMATCH'))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    ok = all([build(n, p, r, a.check) for n, (p, r) in SUBSETS.items()])
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
