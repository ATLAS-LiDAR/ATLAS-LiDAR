#!/usr/bin/env python3
"""Evaluate OpenPCDet-style detectors on Waymo ATLAS configurations: ASR plus the Waymo metrics.

One job per (model, configuration), each in its own single-GPU process. Each
configuration is a directory under --data_root in the OpenPCDet Waymo layout
(ImageSets/val.txt, waymo_processed_data_v0_5_0/<segment>/...) whose infos carry
`spoof_gt`. The repository's own eval_one_epoch produces result.pkl and the standard
Waymo metrics unchanged; ASR is then scored on every frame with a valid spoof_gt, at
IoU 0.7 for Vehicle and 0.5 for Pedestrian/Cyclist, not conditioned on clean detection.

Models come from --manifest (see manifest.example.json) or from a single --cfg_file/--ckpt.
A model's repository defaults to --repo (eval/phim, the Polar-Hierarchical-Mamba submodule).

  python eval/harness/waymo/run_atlas_eval.py --manifest models.json \\
      --data_root /data/waymo_atlas --output_dir eval/output/waymo --gpus 0,1,2,3 --resume
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

CLASS_NAMES = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist'}
TAGS = {'Vehicle': 'VEHICLE', 'Pedestrian': 'PEDESTRIAN', 'Cyclist': 'CYCLIST'}


def iou_threshold(cls):
    return 0.7 if cls == 1 else 0.5


def attack_summary(infos, annos, attack, iou_fn):
    """ASR over frames with a spoof_gt. `iou_fn(pred[N,7], gt[1,7]) -> [N,1]` overlaps."""
    import numpy as np
    if len(infos) != len(annos):
        raise ValueError('Prediction count does not match dataset')
    targets = successes = 0
    seen = set()
    for info, anno in zip(infos, annos):
        fid = info['frame_id']
        if fid != anno['frame_id'] or fid in seen:
            raise ValueError(f'Misaligned or duplicate prediction: {fid}')
        seen.add(fid)
        target = info.get('spoof_gt')
        if target is None:
            continue
        target = np.asarray(target).reshape(-1)
        if target.size < 8 or not np.isfinite(target).all():
            raise ValueError(f'Invalid spoof_gt: {fid}')
        cls = int(target[-1])
        if cls not in CLASS_NAMES:
            raise ValueError(f'Unsupported target class {cls}: {fid}')
        mask = np.asarray(anno['name']) == CLASS_NAMES[cls]
        detected = bool(mask.any()) and float(np.max(iou_fn(np.asarray(anno['boxes_lidar'])[mask, :7],
                                                            target[None, :7]))) >= iou_threshold(cls)
        targets += 1
        successes += int(not detected if attack == 'removal' else detected)
    if targets == 0:
        raise ValueError('No attack targets found; refusing to report an empty ASR')
    return dict(attack_type=attack, asr=successes / targets, num_targets=targets, num_successes=successes)


def flatten(row):
    """Summary row with Waymo metric keys renamed, plus L1/L2 mAP and mAPH over the three classes."""
    item = {k: v for k, v in row.items() if k != 'metrics'}
    metrics = row.get('metrics', {})
    item.update(metrics)
    if 'recall/rcnn_0.3' in metrics:
        item['recall_rcnn_0.3'] = metrics['recall/rcnn_0.3']
    for level in (1, 2):
        for suffix in ('AP', 'APH'):
            values = []
            for name, tag in TAGS.items():
                key = f'OBJECT_TYPE_TYPE_{tag}_LEVEL_{level}/{suffix}'
                if key in metrics:
                    item[f'{name}_L{level}_{suffix}'] = metrics[key]
                    values.append(metrics[key])
            if len(values) == 3:
                item[f'L{level}_m{suffix}'] = sum(values) / 3
    return item


def discover_datasets(data_root):
    return sorted(p.name for p in Path(data_root).iterdir()
                  if (p / 'ImageSets/val.txt').is_file() and (p / 'waymo_processed_data_v0_5_0').is_dir())


def worker(args):
    """Evaluate one model on one configuration; runs inside the model repository's tools/."""
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / 'tools'))
    os.chdir(args.repo / 'tools')
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    from pcdet.utils import common_utils
    from eval_utils.eval_utils import eval_one_epoch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    cfg.LOCAL_RANK = 0
    name = args.datasets[0]
    common_utils.set_random_seed(1024)
    current = copy.deepcopy(cfg)
    current.DATA_CONFIG.DATA_PATH = str(args.data_root / name)
    current.DATA_CONFIG.USE_SHARED_MEMORY = False
    current.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 1
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(output / 'eval.log', rank=0)
    logger.info('Config: %s; checkpoint: %s; ATLAS data: %s', args.cfg_file, args.ckpt, current.DATA_CONFIG.DATA_PATH)
    dataset, loader, _ = build_dataloader(dataset_cfg=current.DATA_CONFIG, class_names=current.CLASS_NAMES,
                                          batch_size=args.batch_size, dist=False, workers=args.workers,
                                          logger=logger, training=False)
    # The Waymo dataset silently skips segments whose metadata is missing: reject partial runs.
    expected = len(dataset.sample_sequence_list)
    loaded = {info['point_cloud']['lidar_sequence'] for info in dataset.infos}
    if not dataset.infos or len(loaded) != expected:
        raise RuntimeError(f'{name}: missing sequence metadata (expected {expected} sequences)')
    model = build_network(model_cfg=current.MODEL, num_class=len(current.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=True)
    model.cuda().eval()
    metrics = eval_one_epoch(current, args, model, loader, 0, logger, dist_test=False, result_dir=output)
    with (output / 'result.pkl').open('rb') as f:
        annos = pickle.load(f)

    def iou_fn(pred, gt):
        return iou3d_nms_utils.boxes_iou3d_gpu(
            torch.as_tensor(pred, dtype=torch.float32, device='cuda').contiguous(),
            torch.as_tensor(gt, dtype=torch.float32, device='cuda').contiguous()).cpu().numpy()
    attack = 'removal' if name.startswith('removal') else 'injection'
    summary = attack_summary(dataset.infos, annos, attack, iou_fn)
    logger.info('ATLAS %s ASR: %s (%d/%d)', attack, summary['asr'], summary['num_successes'], summary['num_targets'])

    def scalar(value):
        return value.item() if hasattr(value, 'item') else value
    common.atomic_json(output / 'metrics.json', metrics, default=scalar)
    common.atomic_json(output / 'attack_summary.json', summary)
    del annos, model, loader, dataset
    gc.collect()


def model_runs(args, parser):
    if args.manifest:
        specs = common.load_manifest(args.manifest, default_repo=args.repo)
        names = args.models or list(specs)
        unknown = [n for n in names if n not in specs]
        if unknown:
            parser.error('Unknown model(s): ' + ', '.join(unknown))
        return [(n, Path(specs[n]['repo']), Path(specs[n]['cfg']), Path(specs[n]['ckpt']),
                 specs[n].get('python', sys.executable)) for n in names]
    if not (args.cfg_file and args.ckpt):
        parser.error('Pass --manifest, or both --cfg_file and --ckpt')
    return [(args.model_name, args.repo, args.cfg_file, args.ckpt, sys.executable)]


def run_parallel(args, runs):
    pool = common.HealthyGpuPool(common.gpu_slots(args.gpus, args.exclude_gpus, args.max_parallel,
                                                  args.slots_per_gpu))
    jobs = []
    for name, repo, cfg_file, ckpt, python in runs:
        files = [cfg_file, ckpt, HERE / 'run_atlas_eval.py', HERE.parent / 'common.py',
                 repo / 'tools/eval_utils/eval_utils.py', repo / 'pcdet/datasets/waymo/waymo_eval.py',
                 repo / 'pcdet/datasets/waymo/waymo_dataset.py']
        signature = dict(files={str(p): common.sha(p) for p in files}, data_root=str(args.data_root),
                         batch_size=args.batch_size)
        for dataset in args.datasets:
            jobs.append((name, repo, cfg_file, ckpt, python, signature, dataset))

    def run_one(job):
        name, repo, cfg_file, ckpt, python, signature, dataset = job
        output = args.output_dir / name / dataset
        output.mkdir(parents=True, exist_ok=True)
        receipt, log = output / 'done.json', output / 'worker.log'
        current = dict(signature, dataset=dataset)
        row = dict(model=name, dataset=dataset, status='failed', log_path=str(log))
        gpu, healthy = None, True
        try:
            if args.resume and receipt.exists():
                old = json.loads(receipt.read_text())
                if old['signature'] != current:
                    raise ValueError('Inputs changed; use a new --output_dir: ' + dataset)
                if old['row']['status'] == 'ok':
                    print('SKIP', name, dataset, flush=True)
                    return old['row']
            gpu = pool.acquire()
            row['gpu'] = gpu
            command = [python, '-u', str(Path(__file__).resolve()), '--_worker', '--repo', str(repo),
                       '--cfg_file', str(cfg_file), '--ckpt', str(ckpt), '--data_root', str(args.data_root),
                       '--output_dir', str(args.output_dir / name), '--datasets', dataset,
                       '--batch_size', str(args.batch_size), '--workers', str(args.workers)]
            if args.save_to_file:
                command.append('--save_to_file')
            env = common.worker_env(os.environ, gpu, args.threads_per_task)
            print('RUN', name, dataset, 'GPU', gpu, 'log=', log, flush=True)
            with log.open('w') as f:
                code = subprocess.run(command, env=env, cwd=repo / 'tools', stdout=f,
                                      stderr=subprocess.STDOUT).returncode
            row['exit_code'] = code
            if code == 0:
                row.update(json.loads((output / 'attack_summary.json').read_text()),
                           metrics=json.loads((output / 'metrics.json').read_text()), status='ok')
            elif common.log_reports_ecc(log):
                healthy = False
                row['error'] = 'Uncorrectable GPU ECC error; GPU slot quarantined'
                print('QUARANTINE GPU', gpu, flush=True)
            common.atomic_json(receipt, dict(signature=current, row=row))
        except Exception as exc:
            row['error'] = str(exc)
        finally:
            if gpu is not None:
                pool.release(gpu, healthy=healthy)
        print(row['status'].upper(), name, dataset, flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=len(pool.available)) as executor:
        for future in as_completed([executor.submit(run_one, job) for job in jobs]):
            rows.append(future.result())
            common.merge_summary(args.output_dir, rows, ('model', 'dataset'), flatten=flatten)
    failures = sum(r['status'] != 'ok' for r in rows)
    print(f'Completed {len(rows) - failures}/{len(jobs)} jobs: {args.output_dir / "summary.csv"}', flush=True)
    if failures:
        sys.exit(1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, help='JSON model manifest (see manifest.example.json)')
    p.add_argument('--models', nargs='+', help='Subset of manifest models (default: all)')
    p.add_argument('--repo', type=Path, default=HERE.parents[1] / 'phim',
                   help='Default model repository (default: eval/phim)')
    p.add_argument('--cfg_file', type=Path, help='Single-model alternative to --manifest')
    p.add_argument('--ckpt', type=Path, help='Single-model alternative to --manifest')
    p.add_argument('--model_name', default='model', help='Row label for --cfg_file/--ckpt')
    p.add_argument('--data_root', type=Path, required=True, help='Directory holding one subdirectory per configuration')
    p.add_argument('--datasets', nargs='+', help='Configurations to evaluate (default: every one under --data_root)')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--save_to_file', action='store_true')
    p.add_argument('--gpus', help='Comma-separated GPU ids (default: CUDA_VISIBLE_DEVICES)')
    p.add_argument('--exclude_gpus', default='')
    p.add_argument('--max_parallel', type=int, help='Use at most this many GPUs')
    p.add_argument('--slots_per_gpu', type=int, default=1, help='Concurrent jobs per GPU')
    p.add_argument('--threads_per_task', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.threads_per_task < 1 or args.slots_per_gpu < 1:
        p.error('Invalid batch_size / workers / threads_per_task / slots_per_gpu')
    for key in ('repo', 'cfg_file', 'ckpt', 'data_root', 'output_dir', 'manifest'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args._worker:
        worker(args)
        return
    args.datasets = args.datasets or discover_datasets(args.data_root)
    if not args.datasets:
        p.error('No ATLAS configurations under ' + str(args.data_root))
    for name in args.datasets:
        data = args.data_root / name
        if not (data / 'ImageSets/val.txt').is_file() or not (data / 'waymo_processed_data_v0_5_0').is_dir():
            p.error(f'Missing ATLAS validation data: {data}')
    runs = model_runs(args, p)
    for name, repo, cfg_file, ckpt, _ in runs:
        for path in (cfg_file, ckpt, repo / 'tools/eval_utils/eval_utils.py'):
            if not path.is_file():
                p.error(f'{name}: missing {path}')
        print(f'{name:24s} {ckpt}', flush=True)
    print(f'Configurations: {" ".join(args.datasets)}; jobs: {len(runs) * len(args.datasets)}', flush=True)
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        p.error('Output exists; use --resume or a new --output_dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_parallel(args, runs)


if __name__ == '__main__':
    main()
