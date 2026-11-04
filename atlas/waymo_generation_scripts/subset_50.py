"""Pull the 50 study segments out of a dataset you already have, instead of regenerating it.

A 195-segment ATLAS split generated with the same generator, arguments and seed (e.g. the attenuated 6 x 195
<split>_full_att_seed42) contains exactly the frames of the 50-segment dataset for those segments, because
every random draw is keyed by seed and segment/frame, not by how many segments are generated. This copies
(or symlinks) dataset/<seg>.pkl, annos/<seg>_a.pkl and metrics/frames/<seg>_frames.csv for the 50 segments
in data/segments_50_seed42.txt and checks the result.

    python subset_50.py <src_dataset_dir> <dst_dataset_dir> [--link] [--segments data/segments_50_seed42.txt]

Use it ONLY when the source was made with the CVPR generators in this folder (tanh intensity fix) and the same
settings as the target name in datasets.sh; NeurIPS-era datasets have the intensity bug and must be regenerated.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path


def put(src, dst, link):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if link:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--segments', default=str(Path(__file__).parent / 'data' / 'segments_50_seed42.txt'))
    ap.add_argument('--link', action='store_true', help='symlink instead of copy (same filesystem only)')
    a = ap.parse_args()
    S, D = Path(a.src), Path(a.dst)
    segs = [l.strip() for l in open(a.segments) if l.strip()]
    missing = [s for s in segs if not (S / 'dataset' / f'{s}.pkl').is_file() or not (S / 'annos' / f'{s}_a.pkl').is_file()]
    if missing:
        sys.exit(f'{len(missing)} of {len(segs)} segments missing in {S} (first: {missing[0]}); regenerate instead')
    for s in segs:
        put(S / 'dataset' / f'{s}.pkl', D / 'dataset' / f'{s}.pkl', a.link)
        put(S / 'annos' / f'{s}_a.pkl', D / 'annos' / f'{s}_a.pkl', a.link)
        f = S / 'metrics' / 'frames' / f'{s}_frames.csv'
        if f.is_file():
            put(f, D / 'metrics' / 'frames' / f.name, a.link)
    m = S / 'metrics' / 'run_manifest.json'
    if m.is_file():
        put(m, D / 'metrics' / 'run_manifest_parent.json', a.link)
    nd = len(list((D / 'dataset').glob('*.pkl'))); na = len(list((D / 'annos').glob('*_a.pkl')))
    ok = nd == na == len(segs)
    print(f'{D}: {nd} dataset / {na} annos from {S}' + (' OK' if ok else ' MISMATCH'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
