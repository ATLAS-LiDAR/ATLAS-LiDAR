#!/usr/bin/env python3
"""Add real-object recall for one class to a run_atlas_eval.py summary.

For a phantom-injection run it answers a question ASR cannot: does the phantom also make
the detector miss the real objects of that class? Recall is read from each job's
metrics_details.json, i.e. the devkit's center-distance matching on the attacked frames
(phantoms are not ground truth), at the devkit's 1% recall resolution:

  recall_<cls>_max        max recall over all predictions, mean of the 0.5/1/2/4 m thresholds
  recall_<cls>_2m_max     max recall at 2 m
  recall_<cls>_2m_s<S>    recall at 2 m, keeping only predictions with score >= S
  recall_<cls>_0.5m_max   max recall at 0.5 m, the strictest threshold
  recall_<cls>_0.5m_s<S>  the same, with the score threshold

  python eval/harness/nuscenes/add_class_recall.py <output_dir> --class pedestrian
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

DISTS = ('0.5', '1.0', '2.0', '4.0')


def recall_at(detail, score):
    # The devkit pads the interpolated curve with zeros past the highest reached recall.
    r, p, c = (np.asarray(detail[k]) for k in ('recall', 'precision', 'confidence'))
    keep = (p > 0) & (c >= score)
    return float(r[keep].max()) if keep.any() else 0.0


def recalls(details, cls, score):
    return {
        f'recall_{cls}_max': float(np.mean([recall_at(details[f'{cls}:{t}'], 0.0) for t in DISTS])),
        f'recall_{cls}_2m_max': recall_at(details[f'{cls}:2.0'], 0.0),
        f'recall_{cls}_2m_s{score}': recall_at(details[f'{cls}:2.0'], score),
        f'recall_{cls}_0.5m_max': recall_at(details[f'{cls}:0.5'], 0.0),
        f'recall_{cls}_0.5m_s{score}': recall_at(details[f'{cls}:0.5'], score),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('output_dir', type=Path)
    p.add_argument('--class', dest='cls', default='pedestrian')
    p.add_argument('--score', type=float, default=0.3)
    args = p.parse_args()
    rows = json.loads((args.output_dir / 'summary.json').read_text())
    updated = 0
    for row in rows:
        details = args.output_dir / row['model'] / row['config'] / 'metrics_details.json'
        if row.get('status') != 'ok' or not details.is_file():
            continue
        row.update(recalls(json.loads(details.read_text()), args.cls, args.score))
        updated += 1
    tmp = args.output_dir / 'summary.json.tmp'
    tmp.write_text(json.dumps(rows, indent=2))
    tmp.replace(args.output_dir / 'summary.json')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = args.output_dir / 'summary.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.output_dir / 'summary.csv')
    print(f'{updated}/{len(rows)} rows updated in {args.output_dir}')


if __name__ == '__main__':
    main()
