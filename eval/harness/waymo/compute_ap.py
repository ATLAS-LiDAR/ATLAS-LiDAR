#!/usr/bin/env python3
"""Waymo L1/L2 AP and APH from saved result.pkl files, without re-running inference.

For runs whose evaluation stopped at ASR (or whose code did not report AP), this scores
<output_dir>/<model>/<configuration>/result.pkl against that configuration's ground
truth with the OpenPCDet Waymo evaluator, writes metrics_ap.json next to each
result.pkl, and collects everything, with ASR where attack_summary.json exists, into
<output_dir>/results_with_ap.csv. Finished metrics_ap.json files are reused.

  python eval/harness/waymo/compute_ap.py --output_dir eval/output/waymo --data_root /data/waymo_atlas
"""
import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']
TAGS = {'Vehicle': 'VEHICLE', 'Pedestrian': 'PEDESTRIAN', 'Cyclist': 'CYCLIST'}


def load_gt(data):
    gt = {}
    for line in (data / 'ImageSets/val.txt').read_text().split():
        seg = line.replace('.tfrecord', '')
        with (data / 'waymo_processed_data_v0_5_0' / seg / f'{seg}.pkl').open('rb') as f:
            for info in pickle.load(f):
                gt[info['frame_id']] = info['annos']
    return gt


def summary_row(model, dataset, ap, attack=None):
    row = dict(model=model, dataset=dataset)
    if attack:
        row.update(asr=attack['asr'], num_targets=attack['num_targets'], num_successes=attack['num_successes'])
    for level in (1, 2):
        for suffix in ('AP', 'APH'):
            values = [ap[f'OBJECT_TYPE_TYPE_{TAGS[c]}_LEVEL_{level}/{suffix}'] for c in CLASSES]
            row.update({f'{c}_L{level}_{suffix}': v for c, v in zip(CLASSES, values)})
            row[f'L{level}_m{suffix}'] = sum(values) / 3
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output_dir', type=Path, required=True, help='Holds <model>/<configuration>/result.pkl')
    p.add_argument('--data_root', type=Path, required=True, help='Holds one subdirectory per configuration')
    p.add_argument('--models', nargs='+', help='Default: every subdirectory of --output_dir')
    p.add_argument('--datasets', nargs='+', help='Default: every configuration with a result.pkl')
    p.add_argument('--repo', type=Path, default=HERE.parents[1] / 'phim',
                   help='OpenPCDet-style repository providing the Waymo evaluator (default: eval/phim)')
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from pcdet.datasets.waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
    models = args.models or sorted(d.name for d in args.output_dir.iterdir() if d.is_dir())
    rows = []
    for model in models:
        datasets = args.datasets or sorted(d.parent.name for d in (args.output_dir / model).glob('*/result.pkl'))
        for name in datasets:
            out = args.output_dir / model / name
            if (out / 'metrics_ap.json').exists():
                ap = json.loads((out / 'metrics_ap.json').read_text())
            else:
                with (out / 'result.pkl').open('rb') as f:
                    det = pickle.load(f)
                gt = load_gt(args.data_root / name)
                if len(det) != len(gt) or any(d['frame_id'] not in gt for d in det):
                    raise ValueError(f'{model}/{name}: predictions do not cover the dataset')
                ap = OpenPCDetWaymoDetectionMetricsEstimator().waymo_evaluation(
                    det, [gt[d['frame_id']] for d in det], class_name=CLASSES,
                    distance_thresh=1000, fake_gt_infos=False)
                ap = {k: float(v[0]) for k, v in ap.items()}
                (out / 'metrics_ap.json').write_text(json.dumps(ap, indent=2))
            attack = out / 'attack_summary.json'
            row = summary_row(model, name, ap, json.loads(attack.read_text()) if attack.exists() else None)
            rows.append(row)
            print(model, name, 'ASR', row.get('asr'), 'L2 mAP', round(row['L2_mAP'], 4),
                  'L2 mAPH', round(row['L2_mAPH'], 4), flush=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output_dir / 'results_with_ap.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows -> {args.output_dir / "results_with_ap.csv"}')


if __name__ == '__main__':
    main()
