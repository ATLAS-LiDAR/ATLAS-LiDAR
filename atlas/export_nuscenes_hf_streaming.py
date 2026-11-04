#!/usr/bin/env python
"""Low-memory front end to export_nuscenes_hf.py.

export_nuscenes_hf.write_split() holds every shard of a split in memory as Python lists
(about 2.3 GB per scene, about 80 GB per split) and writes only once the whole split has
been read. This writes each shard as soon as it is full, so memory is bounded by one
shard. Rows, schema, shard boundaries, file names and the round-trip check are the
exporter's own; only the buffering differs.

  python atlas/export_nuscenes_hf_streaming.py --atlas_dir <atlas_val> --out <dataset_root> \\
      --splits inject_global_pedestrian_easy ... [--shard_gb 0.1]
"""
import argparse
import glob
import math
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_nuscenes_hf as hf  # noqa: E402


def write_split(split, src_root, out_root, target_shard_gb):
    family = 'inject' if split.startswith('inject') else 'removal'
    scheduled = split.endswith(tuple(hf.SCHED_SUFFIXES))
    annos_dir = os.path.join(src_root, split + '_annos')
    out_dir = os.path.join(out_root, 'data', split)
    os.makedirs(out_dir, exist_ok=True)
    schema = hf.build_schema(family, scheduled)
    seg_files = sorted(glob.glob(os.path.join(src_root, split, '*_d.pkl')))
    if not seg_files:
        raise FileNotFoundError(os.path.join(src_root, split))

    # Same segment-count budget as hf.write_split.
    budget = int(max(1, target_shard_gb * 1e9 // (7 * 3.4e6)))
    n = math.ceil(len(seg_files) / budget)
    written, buf, n_rows = [], [], 0

    def flush():
        path = os.path.join(out_dir, f'train-{len(written):05d}-of-{n:05d}.parquet')
        pq.write_table(pa.Table.from_pylist(buf, schema=schema), path, compression='zstd')
        written.append(path)
        buf.clear()

    for i, sf in enumerate(seg_files, 1):
        af = os.path.join(annos_dir, os.path.basename(sf).replace('_d.pkl', '_a.pkl'))
        rows = list(hf.segment_rows(sf, af, family, scheduled))
        buf.extend(rows)
        n_rows += len(rows)
        if i % budget == 0 and buf:
            flush()
    if buf:
        flush()
    if len(written) != n:  # a segment group with no attacked frames would leave a gap in -of-N
        for k, path in enumerate(written):
            os.replace(path, os.path.join(out_dir, f'train-{k:05d}-of-{len(written):05d}.parquet'))
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out_dir, '*.parquet')))
    print(f'  {split:34s} {len(seg_files):4d} segments -> {n_rows:5d} attacked frames, '
          f'{len(written)} shard(s), {size / 1e9:.2f} GB', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--atlas_dir', required=True, help='Generator output: <split>/ and <split>_annos/')
    ap.add_argument('--out', required=True, help='Dataset root; shards go to <out>/data/<split>/')
    ap.add_argument('--splits', nargs='+', required=True)
    ap.add_argument('--shard_gb', type=float, default=0.45)
    a = ap.parse_args()
    print('writing Parquet (ATTACKED frames only):')
    for s in a.splits:
        write_split(s, a.atlas_dir, a.out, a.shard_gb)
    print('\nverifying round-trip:')
    ok = all(hf.verify(s, a.atlas_dir, a.out) for s in a.splits)
    print(f"\n{'ALL ROUND-TRIPS OK' if ok else 'SOME ROUND-TRIPS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
